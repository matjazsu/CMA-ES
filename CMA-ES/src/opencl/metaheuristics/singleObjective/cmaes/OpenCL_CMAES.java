package opencl.metaheuristics.singleObjective.cmaes;

import java.util.Comparator;
import java.util.Random;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;

import static org.jocl.CL.*;
import opencl.helpers.OpenCL_Kernels;
import opencl.helpers.OpenCL_Kernels_Enums;
import opencl.helpers.OpenCL_Manager;
import jmetal.core.Algorithm;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.core.Variable;
import jmetal.metaheuristics.singleObjective.cmaes.Utils;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.comparators.ObjectiveComparator;

/**
 * This class implements the CMA-ES algorithm on OpenCL
 * @author: matjaz suber
 * @date: 02.2015
 */
public class OpenCL_CMAES extends Algorithm {

	/**
	 * @serialVersionUID
	 */
	private static final long serialVersionUID = 1L;

	//Stores the population size
	private int populationSize;

	private int counteval;
	private int maxEvaluations;

	private float sigma;

	private float [] xmean;
	private float [] xold;

	//Strategy parameter setting: Selection
	private int mu;
	private float [] weights;
	private float mueff;

	//Strategy parameter setting: Adaptation
	private float cc;
	private float cs;
	private float c1;
	private float cmu;
	private float damps;

	//Dynamic (internal) strategy parameters and constants
	private float [] pc;
	private float [] ps;
	private float [][] B;
	private float [] diagD;
	private float [][] C;
	private float [][] invsqrtC;
	private int eigeneval;
	private float chiN;

	private float [][] arx;

	//SolutionSet
	private SolutionSet population_;
	//Best solution ever
	private Solution bestSolutionEver = null;

	//OpenCL Kernels variables
	private String samplePopulationKernel = "";
	private String updateDistributionKernel = "";
	private String updateDistributionHelperKernel = "";
	private String resampleSingleKernel = "";

	//Refence to OpenCL_Manager
	private OpenCL_Manager _openCLManager;

	//Reference to Random class
	private Random rand;

	//cl_mem objects declaration
	private cl_mem xoldMem;
	private cl_mem xmeanMem;
	private cl_mem weightsMem;
	private cl_mem arx2arrayMem;
	private cl_mem arindexMem;
	private cl_mem artmpMem;
	private cl_mem invsqrtC2arrayMem;
	private cl_mem psMem;
	private cl_mem diagDMem;
	private cl_mem B2arrayMem;
	private cl_mem arfitnessMem;
	private cl_mem upperLimitMem;
	private cl_mem lowerLimitMem;
	private cl_mem psxpsMem;
	private cl_mem pcMem;
	private cl_mem C2arrayMem;
	private cl_mem artmp22arrayMem;

	//########################### CMAES_OpenCL constructor ###########################//

	//Constructor CMAES_OpenCL
	public OpenCL_CMAES(Problem problem) throws Exception {
		super(problem);

		//Initialize JMetal Random object
		long seed = System.currentTimeMillis();
		rand = new Random(seed);

		//Initialize OpenCL_Manager
		_openCLManager = new OpenCL_Manager();

		//Initializa OpenCL context
		_openCLManager.InitOpenCLContext();

		//Init OpenCL kernels
		initOpenCLKernels();
	}

	/**
	 * Read and set OpenCL Kernels
	 * @throws Exception 
	 */
	private void initOpenCLKernels() throws Exception{
		//Read samplePopulationKernel
		samplePopulationKernel = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_samplePopulation);

		//Read updateDistributionKernel 
		updateDistributionKernel = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_updateDistribution);

		//Read updateDistributionHelperKernel 
		updateDistributionHelperKernel = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_updateDistributionHelper);

		//Read resampleSingleKernel
		resampleSingleKernel = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_resampleSingle);
	}

	//########################### EXECUTE ###########################//

	//########################### Method execute ###########################//

	@Override
	public SolutionSet execute() throws JMException, ClassNotFoundException {

		//Host: Read the parameters
		populationSize = (Integer) getInputParameter("populationSize");
		maxEvaluations = (Integer) getInputParameter("maxEvaluations");

		//Host: Initialize the variables
		counteval = 0;

		//Host: Initalize ObjectiveComparator
		Comparator comparator = new ObjectiveComparator(0);

		//Host: Initialize class variables
		init();

		//Host: main loop
		while(counteval < maxEvaluations){
			//Get a new population of solutions
			population_ = samplePopulation();
			
			for(int i = 0; i < populationSize; i++) {
				if (!isFeasible(population_.get(i))) {
					population_.replace(i, resampleSingle(i));
				}
				problem_.evaluate(population_.get(i));
				counteval += populationSize;
			}

			//Host: stores best solution
			storeBest(comparator);
			System.out.println(counteval + ": " + bestSolutionEver);
			//Host: uses OpenCL to update distribution
			updateDistribution();
		}

		SolutionSet resultPopulation  = new SolutionSet(1);
		resultPopulation.add(bestSolutionEver);

		//Release OpenCL environment
		_openCLManager.ReleaseOpenCLEnvironment();

		return resultPopulation;
	}

	//########################### INIT ###########################//


	//########################### Method init() ###########################//

	/**
	 * Initializes OpenCL_CMAES class variables
	 * CMA-ES original method
	 * @throws Exception 
	 */
	private void init() throws ClassNotFoundException {

		/* User defined input parameters */

		// number of objective variables/problem dimension
		int N = problem_.getNumberOfVariables();

		// objective variables initial point
		xmean = new float[N];
		for (int i = 0; i < N; i++) {
			xmean[i] = (float) PseudoRandom.randDouble(0, 1);
		}

		// coordinate wise standard deviation (step size)
		sigma = (float) 0.3;
		//sigma = 1;

		/* Strategy parameter setting: Selection */

		// population size, offspring number
		int lambda = populationSize;
		//lambda = 4+Math.floor(3*Math.log(N));

		// number of parents/points for recombination
		mu = (int) Math.floor(lambda/2);

		// muXone array for weighted recombination
		weights = new float[mu];
		float sum = 0;
		for (int i=0; i<mu; i++) {
			weights[i] = (float) (Math.log(mu + 1/2) - Math.log(i + 1));
			sum += weights[i];
		}
		// normalize recombination weights array
		for (int i=0; i<mu; i++) {
			weights[i] = weights[i]/sum;
		}

		// variance-effectiveness of sum w_i x_i
		double sum1 = 0;
		double sum2 = 0;
		for (int i = 0; i < mu; i++) {
			sum1 += weights[i];
			sum2 += weights[i] * weights[i];
		}
		mueff = (float) (sum1 * sum1 / sum2);

		/* Strategy parameter setting: Adaptation */

		// time constant for cumulation for C
		cc = (4 + mueff/N) / (N + 4 + 2*mueff/N);

		// t-const for cumulation for sigma control
		cs = (mueff + 2) / (N + mueff + 5);

		// learning rate for rank-one update of C
		c1 = (float) (2 / ((N+1.3)*(N+1.3) + mueff));

		// learning rate for rank-mu update
		cmu = (float) Math.min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((N+2)*(N+2) + mueff));

		// damping for sigma, usually close to 1
		damps = (float) (1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (N+1)) -1) + cs);

		/* Initialize dynamic (internal) strategy parameters and constants */

		// diagonal D defines the scaling
		diagD = new float[N];

		// evolution paths for C and sigma
		pc = new float[N];
		ps = new float[N];

		// B defines the coordinate system
		B  = new float[N][N];
		// covariance matrix C
		C  = new float[N][N];

		// C^-1/2
		invsqrtC  = new float[N][N];

		for (int i = 0; i < N; i++) {
			pc[i] = 0;
			ps[i] = 0;
			diagD[i] = 1;
			for (int j = 0; j < N; j++) {
				B[i][j] = 0;
				invsqrtC[i][j] = 0;
			}
			for (int j = 0; j < i; j++) {
				C[i][j] = 0;
			}
			B[i][i] = 1;
			C[i][i] = diagD[i] * diagD[i];
			invsqrtC[i][i] = 1;
		}

		// track update of B and D
		eigeneval = 0;

		chiN = (float) (Math.sqrt(N) * ( 1 - 1/(4*N) + 1/(21*N*N) ));

		/* non-settable parameters */

		xold = new float[N];
		arx = new float[lambda][N];
	}

	//########################### Method samplePopulation() ###########################//


	/**
	 * OpenCL implementation of the method samplePopulation()
	 * Reason: better performance
	 * @return SolutionSet
	 */
	private SolutionSet samplePopulation() throws JMException, ClassNotFoundException{

		int N = problem_.getNumberOfVariables();
		float [] artmp = new float[N];

		//B2array
		float[] B2array = matrix2array(B);
		//arx2array
		float[] arx2array = matrix2array(arx);

		//Create pointers
		Pointer p_B2array = Pointer.to(B2array);
		Pointer p_arx2array = Pointer.to(arx2array);
		Pointer p_xmean = Pointer.to(xmean);

		//B2array
		B2arrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
				Sizeof.cl_float * B2array.length,
				p_B2array, 
				null);

		//arx2array
		arx2arrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * arx2array.length,
				p_arx2array, 
				null);

		//xmean
		xmeanMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				Sizeof.cl_float * xmean.length,
				p_xmean, 
				null);

		//Create the program from the source code
		_openCLManager.program = clCreateProgramWithSource(_openCLManager.context,
				1, 
				new String[]{ samplePopulationKernel }, 
				null, 
				null);

		//Build the program
		clBuildProgram(_openCLManager.program, 
				0, 
				null, 
				null, 
				null, 
				null);

		//Create the kernel
		_openCLManager.kernel = clCreateKernel(_openCLManager.program, 
				OpenCL_Kernels_Enums.name_samplePopulation, 
				null);

		//Set the arguments for the kernel
		clSetKernelArg(_openCLManager.kernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ N }));
		clSetKernelArg(_openCLManager.kernel, 2, Sizeof.cl_mem, Pointer.to(B2arrayMem));
		clSetKernelArg(_openCLManager.kernel, 3, Sizeof.cl_mem, Pointer.to(arx2arrayMem));
		clSetKernelArg(_openCLManager.kernel, 4, Sizeof.cl_mem, Pointer.to(xmeanMem));
		clSetKernelArg(_openCLManager.kernel, 5, Sizeof.cl_float, Pointer.to(new float[]{ sigma }));

		//Set local_work_size
		long localWorkSize = _openCLManager.getLocalWorkSize(N);
		//Set global_work_size
		long globalWorkSize = _openCLManager.getGlobalWorkSize(localWorkSize, N);

		for (int iNk = 0; iNk < populationSize; iNk++) {

			for (int i = 0; i < N; i++) {
				//TODO: Check the correctness of this random (http://en.wikipedia.org/wiki/CMA-ES)
				artmp[i] = (float) (diagD[i] * rand.nextGaussian());
			}

			Pointer p_artmp = Pointer.to(artmp);

			//artmp
			artmpMem = clCreateBuffer(_openCLManager.context, 
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
					Sizeof.cl_float * artmp.length,
					p_artmp, 
					null);

			//Set the arguments for the kernel
			clSetKernelArg(_openCLManager.kernel, 1, Sizeof.cl_mem, Pointer.to(artmpMem));
			clSetKernelArg(_openCLManager.kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{ iNk }));

			//Execute the kernel
			clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
					_openCLManager.kernel, 
					1, 
					null,
					new long[]{ globalWorkSize }, 
					new long[]{ localWorkSize }, 
					0, 
					null, 
					null);

		}		

		//Read the output data
		clEnqueueReadBuffer(_openCLManager.commandQueue, 
				arx2arrayMem, 
				CL_TRUE, 
				0,
				Sizeof.cl_float * arx2array.length, 
				p_arx2array, 
				0, 
				null, 
				null);

		//Wait until all commands has finished in the command queue
		clFinish(_openCLManager.commandQueue);
		//Release kernel
		clReleaseKernel(_openCLManager.kernel);
		//Release program
		clReleaseProgram(_openCLManager.program);

		arx = array2matrix(arx2array, arx.length, arx[0].length);

		return genoPhenoTransformation(arx);
	}

	//########################### Method updateDistribution() ###########################//

	/**
	 * OpenCL implementation of the method updateDistribution()
	 * Reason: better performance
	 * @throws JMException 
	 */
	private void updateDistribution() throws JMException{

		//Local variables
		int N = problem_.getNumberOfVariables();
		int lambda = populationSize;
		float [] arfitness = new float[lambda];
		int [] arindex = new int[lambda];

		//minimization
		for (int i = 0; i < lambda; i++) {
			arfitness[i] = (float)population_.get(i).getObjective(0);
			arindex[i] = i;
		}
		Utils.minFastSort(arfitness, arindex, lambda);

		//########################### OpenCL ###########################//

		//ps
		Pointer p_ps = Pointer.to(ps);
		psMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * ps.length,
				p_ps, 
				null);

		//psxps
		float[] psxps = new float[N];
		Pointer p_psxps = Pointer.to(psxps);
		psxpsMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * psxps.length,
				p_psxps, 
				null);

		//arx2array
		float[] arx2array = matrix2array(arx);
		Pointer p_arx2array = Pointer.to(arx2array);
		arx2arrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * arx2array.length,
				p_arx2array, 
				null);

		//arindex
		Pointer p_arindex = Pointer.to(arindex);
		arindexMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * arindex.length,
				p_arindex, 
				null);

		//xmean
		Pointer p_xmean = Pointer.to(xmean);
		xmeanMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * xmean.length,
				p_xmean, 
				null);

		//xold
		Pointer p_xold = Pointer.to(xold);
		xoldMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * xold.length,
				p_xold, 
				null);

		//pc
		Pointer p_pc = Pointer.to(pc);
		pcMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * pc.length,
				p_pc, 
				null);

		//artmp
		float[] artmp = new float[N];
		Pointer p_artmp = Pointer.to(artmp);
		artmpMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * artmp.length,
				p_artmp, 
				null);

		//invsqrtC2array
		float[] invsqrtC2array = matrix2array(invsqrtC);
		Pointer p_invsqrtC2array = Pointer.to(invsqrtC2array);
		invsqrtC2arrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * invsqrtC2array.length,
				p_invsqrtC2array, 
				null);

		//C2array
		float[] C2array = matrix2array(C);
		Pointer p_C2array = Pointer.to(C2array);
		C2arrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * C2array.length,
				p_C2array, 
				null);

		//weights
		Pointer p_weights = Pointer.to(weights);
		weightsMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * weights.length,
				p_weights, 
				null);

		//Create the program from the source code
		_openCLManager.program = clCreateProgramWithSource(_openCLManager.context,
				1, 
				new String[]{ updateDistributionKernel }, 
				null, 
				null);

		//Build the program
		clBuildProgram(_openCLManager.program, 
				0, 
				null, 
				null, 
				null, 
				null);

		//Create the kernel
		_openCLManager.kernel = clCreateKernel(_openCLManager.program, 
				OpenCL_Kernels_Enums.name_updateDistribution, 
				null);

		//Set the arguments for the kernel
		clSetKernelArg(_openCLManager.kernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ N }));
		clSetKernelArg(_openCLManager.kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{ mu }));
		clSetKernelArg(_openCLManager.kernel, 2, Sizeof.cl_float, Pointer.to(new float[]{ cs }));
		clSetKernelArg(_openCLManager.kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{ counteval }));
		clSetKernelArg(_openCLManager.kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{ lambda }));
		clSetKernelArg(_openCLManager.kernel, 5, Sizeof.cl_float, Pointer.to(new float[]{ sigma }));
		clSetKernelArg(_openCLManager.kernel, 6, Sizeof.cl_float, Pointer.to(new float[]{ chiN }));
		clSetKernelArg(_openCLManager.kernel, 7, Sizeof.cl_float, Pointer.to(new float[]{ cc }));
		clSetKernelArg(_openCLManager.kernel, 8, Sizeof.cl_float, Pointer.to(new float[]{ mueff }));
		clSetKernelArg(_openCLManager.kernel, 9, Sizeof.cl_float, Pointer.to(new float[]{ cmu }));
		clSetKernelArg(_openCLManager.kernel, 10, Sizeof.cl_float, Pointer.to(new float[]{ c1 }));

		clSetKernelArg(_openCLManager.kernel, 11, Sizeof.cl_mem, Pointer.to(psMem));
		clSetKernelArg(_openCLManager.kernel, 12, Sizeof.cl_float * ps.length, null);
		clSetKernelArg(_openCLManager.kernel, 13, Sizeof.cl_mem, Pointer.to(psxpsMem));
		clSetKernelArg(_openCLManager.kernel, 14, Sizeof.cl_mem, Pointer.to(arx2arrayMem));
		clSetKernelArg(_openCLManager.kernel, 15, Sizeof.cl_mem, Pointer.to(arindexMem));
		clSetKernelArg(_openCLManager.kernel, 16, Sizeof.cl_mem, Pointer.to(xmeanMem));
		clSetKernelArg(_openCLManager.kernel, 17, Sizeof.cl_mem, Pointer.to(xoldMem));
		clSetKernelArg(_openCLManager.kernel, 18, Sizeof.cl_mem, Pointer.to(pcMem));
		clSetKernelArg(_openCLManager.kernel, 19, Sizeof.cl_mem, Pointer.to(artmpMem));
		clSetKernelArg(_openCLManager.kernel, 20, Sizeof.cl_mem, Pointer.to(invsqrtC2arrayMem));
		clSetKernelArg(_openCLManager.kernel, 21, Sizeof.cl_mem, Pointer.to(C2arrayMem));
		clSetKernelArg(_openCLManager.kernel, 22, Sizeof.cl_mem, Pointer.to(weightsMem));

		//Set local_work_size
		long localWorkSize = _openCLManager.getLocalWorkSize(N);
		//Set global_work_size
		long globalWorkSize = _openCLManager.getGlobalWorkSize(localWorkSize, N);

		//Execute the kernel
		clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
				_openCLManager.kernel, 
				1, 
				null,
				new long[]{ globalWorkSize }, 
				new long[]{ localWorkSize }, 
				0, 
				null, 
				null);

		//Read ps
		clEnqueueReadBuffer(_openCLManager.commandQueue, 
				psMem, 
				CL_TRUE, 
				0,
				Sizeof.cl_float * ps.length, 
				p_ps, 
				0, 
				null, 
				null);

		//Read psxps
		clEnqueueReadBuffer(_openCLManager.commandQueue, 
				psxpsMem, 
				CL_TRUE, 
				0,
				Sizeof.cl_float * psxps.length, 
				p_psxps, 
				0, 
				null, 
				null);

		//Read xmean
		clEnqueueReadBuffer(_openCLManager.commandQueue, 
				xmeanMem, 
				CL_TRUE, 
				0,
				Sizeof.cl_float * xmean.length, 
				p_xmean, 
				0, 
				null, 
				null);

		//Read xold
		clEnqueueReadBuffer(_openCLManager.commandQueue, 
				xoldMem, 
				CL_TRUE, 
				0,
				Sizeof.cl_float * xold.length, 
				p_xold, 
				0, 
				null, 
				null);

		//Read pc
		clEnqueueReadBuffer(_openCLManager.commandQueue, 
				pcMem, 
				CL_TRUE, 
				0,
				Sizeof.cl_float * pc.length, 
				p_pc, 
				0, 
				null, 
				null);

		//Read invsqrtC
		clEnqueueReadBuffer(_openCLManager.commandQueue, 
				invsqrtC2arrayMem, 
				CL_TRUE, 
				0,
				Sizeof.cl_float * invsqrtC2array.length, 
				p_invsqrtC2array, 
				0, 
				null, 
				null);

		//Read C
		clEnqueueReadBuffer(_openCLManager.commandQueue, 
				C2arrayMem, 
				CL_TRUE, 
				0,
				Sizeof.cl_float * C2array.length, 
				p_C2array, 
				0, 
				null, 
				null);

		//Wait until all commands has finished in the command queue
		clFinish(_openCLManager.commandQueue);
		//Release kernel
		clReleaseKernel(_openCLManager.kernel);
		//Release program
		clReleaseProgram(_openCLManager.program);

		invsqrtC = array2matrix(invsqrtC2array, invsqrtC.length, invsqrtC[0].length);
		C = array2matrix(C2array, C.length, C[0].length);

		//Adapt step size sigma
		sigma *= Math.exp((cs/damps) * (Math.sqrt(psxps[0])/chiN - 1));

		if (counteval - eigeneval > lambda /(c1+cmu)/N/10) {

			eigeneval = counteval;

			// enforce symmetry
			for (int i = 0; i < N; i++) {
				for (int j = 0; j <= i; j++) {
					B[i][j] = B[j][i] = C[i][j];
				}
			}

			// eigen decomposition, B==normalized eigenvectors
			float [] offdiag = new float[N];
			Utils.tred2(N, B, diagD, offdiag);
			Utils.tql2(N, diagD, offdiag, B);

			//artmp2
			float[][] artmp2 = new float[N][N];
			float[] artmp22array = matrix2array(artmp2);
			Pointer p_artmp22array = Pointer.to(artmp22array);
			artmp22arrayMem = clCreateBuffer(_openCLManager.context, 
					CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
					Sizeof.cl_float * artmp22array.length,
					p_artmp22array, 
					null);

			//B2array
			float[] B2array = matrix2array(B);
			Pointer p_B2array = Pointer.to(B2array);
			B2arrayMem = clCreateBuffer(_openCLManager.context, 
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
					Sizeof.cl_float * B2array.length,
					p_B2array, 
					null);

			//diagD
			Pointer p_diagD = Pointer.to(diagD);
			diagDMem = clCreateBuffer(_openCLManager.context, 
					CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
					Sizeof.cl_float * diagD.length,
					p_diagD, 
					null);

			//invsqrtC2array
			invsqrtC2array = matrix2array(invsqrtC);
			p_invsqrtC2array = Pointer.to(invsqrtC2array);
			invsqrtC2arrayMem = clCreateBuffer(_openCLManager.context, 
					CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
					Sizeof.cl_float * invsqrtC2array.length,
					p_invsqrtC2array, 
					null);

			//Create the program from the source code
			_openCLManager.program = clCreateProgramWithSource(_openCLManager.context,
					1, 
					new String[]{ updateDistributionHelperKernel }, 
					null, 
					null);

			//Build the program
			clBuildProgram(_openCLManager.program, 
					0, 
					null, 
					null, 
					null, 
					null);

			//Create the kernel
			_openCLManager.kernel = clCreateKernel(_openCLManager.program, 
					OpenCL_Kernels_Enums.name_updateDistributionHelper, 
					null);

			clSetKernelArg(_openCLManager.kernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ N }));
			clSetKernelArg(_openCLManager.kernel, 1, Sizeof.cl_mem, Pointer.to(artmp22arrayMem));
			clSetKernelArg(_openCLManager.kernel, 2, Sizeof.cl_mem, Pointer.to(B2arrayMem));
			clSetKernelArg(_openCLManager.kernel, 3, Sizeof.cl_mem, Pointer.to(diagDMem));
			clSetKernelArg(_openCLManager.kernel, 4, Sizeof.cl_mem, Pointer.to(invsqrtC2arrayMem));

			//Execute the kernel
			clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
					_openCLManager.kernel, 
					1, 
					null,
					new long[]{ globalWorkSize }, 
					new long[]{ localWorkSize }, 
					0, 
					null, 
					null);

			//Read diagD
			clEnqueueReadBuffer(_openCLManager.commandQueue, 
					diagDMem, 
					CL_TRUE, 
					0,
					Sizeof.cl_float * diagD.length, 
					p_diagD, 
					0, 
					null, 
					null);

			//Read invsqrtC
			clEnqueueReadBuffer(_openCLManager.commandQueue, 
					invsqrtC2arrayMem, 
					CL_TRUE, 
					0,
					Sizeof.cl_float * invsqrtC2array.length, 
					p_invsqrtC2array, 
					0, 
					null, 
					null);

			//Wait until all commands has finished in the command queue
			clFinish(_openCLManager.commandQueue);
			//Release kernel
			clReleaseKernel(_openCLManager.kernel);
			//Release program
			clReleaseProgram(_openCLManager.program);

			invsqrtC = array2matrix(invsqrtC2array, invsqrtC.length, invsqrtC[0].length);
		}
	}

	//########################### CMA-ES origin methods ###########################//

	/**
	 * Method - isFeasible
	 * @param solution
	 * @return boolean
	 * @throws JMException
	 */
	private boolean isFeasible(Solution solution) throws JMException {
		boolean res = true;
		Variable[] x = solution.getDecisionVariables();

		for (int i = 0; i < problem_.getNumberOfVariables(); i++) {
			double value = x[i].getValue();
			if ((value < problem_.getLowerLimit(i)) || (value > problem_.getUpperLimit(i))) {
				res = false;
				break; //added for optimization
			}
		}
		return res;
	}

	/**
	 * Method - resampleSingle
	 * @param iNk
	 * @return
	 * @throws JMException
	 * @throws ClassNotFoundException
	 */
	private Solution resampleSingle(int iNk) throws JMException, ClassNotFoundException {

		int N = problem_.getNumberOfVariables();

		//Temporary variables
		double[] upperLimit = problem_.getUpperLimit();
		double[] lowerLimit = problem_.getLowerLimit();
		float[] arx2array = matrix2array(arx);

		//Create pointers
		Pointer p_upperLimit = Pointer.to(upperLimit);
		Pointer p_lowerLimit = Pointer.to(lowerLimit);
		Pointer p_arx2array = Pointer.to(arx2array);

		//upperLimit
		upperLimitMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * upperLimit.length,
				p_upperLimit, 
				null);

		//lowerLimit
		lowerLimitMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * lowerLimit.length,
				p_lowerLimit, 
				null);

		//arx2array
		arx2arrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * arx2array.length,
				p_arx2array, 
				null);

		//Create the program from the source code
		_openCLManager.program = clCreateProgramWithSource(_openCLManager.context,
				1, 
				new String[]{ resampleSingleKernel }, 
				null, 
				null);

		//Build the program
		clBuildProgram(_openCLManager.program, 
				0, 
				null, 
				null, 
				null, 
				null);

		//Create the kernel
		_openCLManager.kernel = clCreateKernel(_openCLManager.program, 
				OpenCL_Kernels_Enums.name_resampleSingle, 
				null);

		//Set the arguments for the kernel
		clSetKernelArg(_openCLManager.kernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ N }));
		clSetKernelArg(_openCLManager.kernel, 1, Sizeof.cl_mem, Pointer.to(upperLimitMem));
		clSetKernelArg(_openCLManager.kernel, 2, Sizeof.cl_mem, Pointer.to(lowerLimitMem));
		clSetKernelArg(_openCLManager.kernel, 3, Sizeof.cl_mem, Pointer.to(arx2arrayMem));
		clSetKernelArg(_openCLManager.kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{ iNk }));

		//Set local_work_size
		long localWorkSize = _openCLManager.getLocalWorkSize(N);
		//Set global_work_size
		long globalWorkSize = _openCLManager.getGlobalWorkSize(localWorkSize, N);

		//Execute the kernel
		clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
				_openCLManager.kernel, 
				1, 
				null,
				new long[]{ globalWorkSize }, 
				new long[]{ localWorkSize }, 
				0, 
				null, 
				null);

		//Read the output data
		clEnqueueReadBuffer(_openCLManager.commandQueue, 
				arx2arrayMem, 
				CL_TRUE, 
				0,
				Sizeof.cl_float * arx2array.length, 
				p_arx2array, 
				0, 
				null, 
				null);

		//Wait until all commands has finished in the command queue
		clFinish(_openCLManager.commandQueue);
		//Release kernel
		clReleaseKernel(_openCLManager.kernel);
		//Release program
		clReleaseProgram(_openCLManager.program);

		arx = array2matrix(arx2array, arx.length, arx[0].length);

		return genoPhenoTransformation(arx[iNk]);
	}

	/**
	 * Method - genoPhenoTransformation
	 * @param popx
	 * @return
	 * @throws JMException
	 * @throws ClassNotFoundException
	 */
	private SolutionSet genoPhenoTransformation(float [][] popx) throws JMException, ClassNotFoundException {

		SolutionSet population_ = new SolutionSet(populationSize);
		for (int i = 0; i < populationSize; i++) {
			Solution solution = new Solution(problem_);
			for (int j = 0; j < problem_.getNumberOfVariables(); j++) {
				solution.getDecisionVariables()[j].setValue(popx[i][j]);
			}
			population_.add(solution);
		}
		return population_;
	}

	/**
	 * Method - genoPhenoTransformation
	 * @param x
	 * @return
	 * @throws JMException
	 * @throws ClassNotFoundException
	 */
	private Solution genoPhenoTransformation(float[] x) throws JMException, ClassNotFoundException {
		Solution solution = new Solution(problem_);
		for (int i = 0; i < problem_.getNumberOfVariables(); i++) {
			solution.getDecisionVariables()[i].setValue(x[i]);
		}
		return solution;
	}

	/**
	 * Method - storeBest
	 * @param comparator
	 */
	private void storeBest(Comparator comparator) {
		Solution bestInPopulation = new Solution(population_.best(comparator));
		if ((bestSolutionEver == null) || (bestSolutionEver.getObjective(0) > bestInPopulation.getObjective(0))) {
			bestSolutionEver = bestInPopulation;
		}
	}

	//########################### Private (Helper) methods ###########################//

	/**
	 * Method convert two dimensional array to one dimensional array
	 * @param matrix
	 * @return
	 */
	private float[] matrix2array(float[][] matrix) {
		// TODO Auto-generated method stub

		float[] result = new float[matrix.length * matrix[0].length];
		int index = 0;

		for(int i = 0; i < matrix.length; i++){
			for(int j = 0; j < matrix[0].length; j++){
				result[index] = matrix[i][j];
				index++;
			}
		}

		return result;
	}

	/**
	 * Method convert one dimensional array to two dimensional array
	 * @param array
	 * @param width
	 * @param height
	 * @return
	 */
	private float[][] array2matrix(float[] array, int width, int height){
		// TODO Auto-generated method stub

		float[][] result = new float[width][height];

		int index = 0;

		for(int x = 0; x < width; x ++){
			for(int y = 0; y < height; y++){
				result[x][y] = array[index];
				index++;
			}
		}

		return result;
	}
}
