package opencl.metaheuristics.singleObjective.cmaes;

import java.util.Comparator;
import java.util.Random;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

import static org.jocl.CL.*;
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
	private int N;
	private int lambda;

	private int counteval;
	private int maxEvaluations;

	private float sigma;

	private float [] xmean;
	private float [] xold;
	private float[] psxps;
	private float[] artmp;
	private float[] artmp22array;
	
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
	private float[] B2array;
	private float [] diagD;
	private float [][] C;
	private float [][] invsqrtC;
	private float[] invsqrtC2array;
	private int eigeneval;
	private float chiN;

	private float [][] arx;
	private float[] arx2array;
	
	private float [] arfitness;
	private int [] arindex;
	private float[] randomArray;
	private float[] sumEval;

	//SolutionSet
	private SolutionSet population_;
	//Best solution ever
	private Solution bestSolutionEver = null;
	private float[] bestSolutionEverArray;

	//Refence to OpenCL_Manager
	private OpenCL_Manager _openCLManager;
	
	//Random
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
	private cl_mem psxpsMem;
	private cl_mem pcMem;
	private cl_mem C2arrayMem;
	private cl_mem artmp22arrayMem;
	private cl_mem offdiagMem;
	private cl_mem bestSolutionEverArrayMem;
	private cl_mem arfitnessMem;
	private cl_mem randomArrayMem;
	private cl_mem sumEvalMem;
	
	//cl_pointer 
	private Pointer p_bestSolutionEverArray;

	//########################### CMAES_OpenCL constructor ###########################//

	//Constructor CMAES_OpenCL
	public OpenCL_CMAES(Problem problem) throws Exception {
		super(problem);
		
		//Init seed
		long seed = System.currentTimeMillis();
		
		//Init random
		rand = new Random(seed);

		//Initialize OpenCL_Manager
		_openCLManager = new OpenCL_Manager();

		//Initializa OpenCL context
		_openCLManager.InitOpenCLContext();
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
		
		//Host: Initialize class variables
		init();

		//Host: main loop
		while(counteval < maxEvaluations){
			//Get a new population of solutions
			samplePopulation();
			
			//Update counteval
			counteval += (populationSize*populationSize);
			
			//Host: uses OpenCL to update distribution
			if(counteval < maxEvaluations){
				updateDistribution();
			}
		}
		
		//Read the best solution ever from the device
		clEnqueueReadBuffer(_openCLManager.commandQueue, 
				bestSolutionEverArrayMem, 
				CL_TRUE, 
				0, 
				Sizeof.cl_float * bestSolutionEverArray.length, 
				p_bestSolutionEverArray, 
				0, 
				null, 
				null);
		
		bestSolutionEver = getSolutionFromArray(bestSolutionEverArray);

		SolutionSet resultPopulation  = new SolutionSet(1);
		resultPopulation.add(bestSolutionEver);
		
		//Print best solution ever
		System.out.println("Best solution: " + bestSolutionEver);
		
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

		//number of objective variables/problem dimension
		N = problem_.getNumberOfVariables();
		
		//init lambda
		lambda = populationSize;
		
		//init psxps
		psxps = new float[N];
		
		//init artmp
		artmp = new float[N];
		
		//init artmp22array
		artmp22array = new float[N*N];
		
		//init bestSolutionEverArray
		bestSolutionEverArray = new float[N + 1]; //+1 for evaluation
		
		// objective variables initial point
		xmean = new float[N];
		for (int i = 0; i < N; i++) {
			xmean[i] = (float) PseudoRandom.randDouble(0, 1);
		}

		// coordinate wise standard deviation (step size)
		sigma = (float) 0.3;
		
		//init arfitness
		arfitness = new float[lambda];
		
		//init arindex
		arindex = new int[lambda];
		
		//init random
		randomArray = new float[N];
		for(int i = 0; i < N; i++){
			randomArray[i] = (float) rand.nextGaussian(); 
		}
		
		//init sumEval
		sumEval = new float[1];

		/* Strategy parameter setting: Selection */

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
		
		//initialize memory objects
		initMemoryObjects();
		
		//initialize kernel arguments
		setKernelArgs();
	}

	//########################### Method samplePopulation() ###########################//

	/**
	 * OpenCL implementation of the method samplePopulation()
	 * Reason: better performance
	 */
	private void samplePopulation() throws JMException, ClassNotFoundException{

		//Set local_work_size
		long localWorkSizeSamplePopulation = _openCLManager.getLocalWorkSize(populationSize);
		//Set global_work_size
		long globalWorkSizeSamplePopulation = _openCLManager.getGlobalWorkSize(localWorkSizeSamplePopulation, populationSize);
		
		//Execute the kernel
		clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
				_openCLManager._samplePopulationKernel, 
				1, 
				null,
				new long[]{ globalWorkSizeSamplePopulation }, 
				new long[]{ localWorkSizeSamplePopulation }, 
				0, 
				null, 
				null);
		
		//Set local_work_size
		long localWorkSizeStoreBest = _openCLManager.getLocalWorkSize(N);
		//Set global_work_size
		long globalWorkSizeStoreBest = _openCLManager.getGlobalWorkSize(localWorkSizeStoreBest, N);
		
		//Execute the kernel
		clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
				_openCLManager._storeBestKernel, 
				1, 
				null,
				new long[]{ globalWorkSizeStoreBest }, 
				new long[]{ localWorkSizeStoreBest }, 
				0, 
				null, 
				null);
	}
	
	//########################### Method updateDistribution() ###########################//

	/**
	 * OpenCL implementation of the method updateDistribution()
	 * Reason: better performance
	 * @throws JMException 
	 */
	private void updateDistribution() throws JMException{

		//Set local_work_size
		long localWorkSizeComputeFitness = _openCLManager.getLocalWorkSize(populationSize);
		//Set global_work_size
		long globalWorkSizeComputeFitness = _openCLManager.getGlobalWorkSize(localWorkSizeComputeFitness, populationSize);
		
		//Execute the kernel
		clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
				_openCLManager._computeFitnessKernel, 
				1, 
				null,
				new long[]{ globalWorkSizeComputeFitness }, 
				new long[]{ localWorkSizeComputeFitness }, 
				0, 
				null, 
				null);

		//Set local_work_size
		long localWorkSizeCalculateXmean = _openCLManager.getLocalWorkSize(N);
		//Set global_work_size
		long globalWorkSizeCalculateXmean = _openCLManager.getGlobalWorkSize(localWorkSizeCalculateXmean, N);
		
		//Execute the kernel
		clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
				_openCLManager._calculateXmeanKernel, 
				1, 
				null,
				new long[]{ globalWorkSizeCalculateXmean }, 
				new long[]{ localWorkSizeCalculateXmean }, 
				0, 
				null, 
				null);
		
		//Set local_work_size
		long localWorkSizeUpdateEvolutionPathsKernel = _openCLManager.getLocalWorkSize(N);
		//Set global_work_size
		long globalWorkSizeUpdateEvolutionPathsKernel = _openCLManager.getGlobalWorkSize(localWorkSizeUpdateEvolutionPathsKernel, N);
		
		//Execute the kernel
		clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
				_openCLManager._updateEvolutionPathsKernel, 
				1, 
				null,
				new long[]{ globalWorkSizeUpdateEvolutionPathsKernel }, 
				new long[]{ localWorkSizeUpdateEvolutionPathsKernel }, 
				0, 
				null, 
				null);
		
		//Set local_work_size
		long localWorkSizeCalculateNormKernel = _openCLManager.getLocalWorkSize(N);
		//Set global_work_size
		long globalWorkSizeCalculateNormKernel = _openCLManager.getGlobalWorkSize(localWorkSizeCalculateNormKernel, N);
		
		//Execute the kernel
		clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
				_openCLManager._calculateNormKernel, 
				1, 
				null,
				new long[]{ globalWorkSizeCalculateNormKernel }, 
				new long[]{ localWorkSizeCalculateNormKernel }, 
				0, 
				null, 
				null);
		
		//Set local_work_size
		long localWorkSizeAdaptCovarianceMatrixKernel = _openCLManager.getLocalWorkSize(N);
		//Set global_work_size
		long globalWorkSizeAdaptCovarianceMatrixKernel = _openCLManager.getGlobalWorkSize(localWorkSizeAdaptCovarianceMatrixKernel, N);
		
		//Execute the kernel
		clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
				_openCLManager._adaptCovarianceMatrixKernel, 
				1, 
				null,
				new long[]{ globalWorkSizeAdaptCovarianceMatrixKernel }, 
				new long[]{ localWorkSizeAdaptCovarianceMatrixKernel }, 
				0, 
				null, 
				null);
		
		if (counteval - eigeneval > lambda /(c1+cmu)/N/10) {

			//set eigeneval
			eigeneval = counteval;
			
			//Set local_work_size
			long localWorkSizeUpdateDistributionHelperKernel = _openCLManager.getLocalWorkSize(N);
			//Set global_work_size
			long globalWorkSizeUpdateDistributionHelperKernel = _openCLManager.getGlobalWorkSize(localWorkSizeUpdateDistributionHelperKernel, N);

			//Execute the kernel
			clEnqueueNDRangeKernel(_openCLManager.commandQueue, 
					_openCLManager._updateDistributionHelperKernel, 
					1, 
					null,
					new long[]{ globalWorkSizeUpdateDistributionHelperKernel }, 
					new long[]{ localWorkSizeUpdateDistributionHelperKernel }, 
					0, 
					null, 
					null);
		}
	}

	//########################### Method initStaticKernelArgs() ###########################//
	
	/**
	 * Initialize memory object for OpenCL
	 */
	private void initMemoryObjects(){
		
		//diagD
		Pointer p_diagD = Pointer.to(diagD);
		diagDMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * diagD.length,
				p_diagD, 
				null);
		
		//B2array
		B2array = matrix2array(B);
		Pointer p_B2array = Pointer.to(B2array);
		B2arrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
				Sizeof.cl_float * B2array.length,
				p_B2array, 
				null);
		
		//arx2array
		arx2array = matrix2array(arx);
		Pointer p_arx2array = Pointer.to(arx2array);
		arx2arrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * arx2array.length,
				p_arx2array, 
				null);
		
		//xmean
		Pointer p_xmean = Pointer.to(xmean);
		xmeanMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				Sizeof.cl_float * xmean.length,
				p_xmean, 
				null);
		
		//randomArray
		Pointer p_randomArray = Pointer.to(randomArray);
		randomArrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				Sizeof.cl_float * randomArray.length,
				p_randomArray, 
				null);
		
		//bestSolutionEverArray
		p_bestSolutionEverArray = Pointer.to(bestSolutionEverArray);
		bestSolutionEverArrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * bestSolutionEverArray.length,
				p_bestSolutionEverArray, 
				null);
		
		Pointer p_sumEval = Pointer.to(sumEval);
		sumEvalMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * sumEval.length,
				p_sumEval, 
				null);
		
		//arfitness
		Pointer p_arfitness = Pointer.to(arfitness);
		arfitnessMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * arfitness.length,
				p_arfitness, 
				null);
		
		//arindex
		Pointer p_arindex = Pointer.to(arindex);
		arindexMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * arindex.length,
				p_arindex, 
				null);
		
		//ps
		Pointer p_ps = Pointer.to(ps);
		psMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * ps.length,
				p_ps, 
				null);
		
		//psxps
		Pointer p_psxps = Pointer.to(psxps);
		psxpsMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * psxps.length,
				p_psxps, 
				null);
		
		//pc
		Pointer p_pc = Pointer.to(pc);
		pcMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * pc.length,
				p_pc, 
				null);
		
		//artmp
		Pointer p_artmp = Pointer.to(artmp);
		artmpMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * artmp.length,
				p_artmp, 
				null);
		
		//xold
		Pointer p_xold = Pointer.to(xold);
		xoldMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * xold.length,
				p_xold, 
				null);
		
		//weights
		Pointer p_weights = Pointer.to(weights);
		weightsMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * weights.length,
				p_weights, 
				null);
		
		//C2array
		float[] C2array = matrix2array(C);
		Pointer p_C2array = Pointer.to(C2array);
		C2arrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * C2array.length,
				p_C2array, 
				null);
		
		//invsqrtC2array
		invsqrtC2array = matrix2array(invsqrtC);
		Pointer p_invsqrtC2array = Pointer.to(invsqrtC2array);
		invsqrtC2arrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * invsqrtC2array.length,
				p_invsqrtC2array, 
				null);
		
		
		
		//artmp2
		Pointer p_artmp22array = Pointer.to(artmp22array);
		artmp22arrayMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * artmp22array.length,
				p_artmp22array, 
				null);
		
		//offdiag
		float [] offdiag = new float[N];
		Pointer p_offdiag = Pointer.to(offdiag);
		offdiagMem = clCreateBuffer(_openCLManager.context, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				Sizeof.cl_float * offdiag.length,
				p_offdiag, 
				null);
	}
	
	/**
	 * Set static OpenCL Kernel's arguments
	 */
	private void setKernelArgs(){
		
		//Kernel - samplePopulationKernel
		clSetKernelArg(_openCLManager._samplePopulationKernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ populationSize }));
		clSetKernelArg(_openCLManager._samplePopulationKernel, 1, Sizeof.cl_int, Pointer.to(new int[]{ N }));
		clSetKernelArg(_openCLManager._samplePopulationKernel, 2, Sizeof.cl_mem, Pointer.to(diagDMem));
		clSetKernelArg(_openCLManager._samplePopulationKernel, 3, Sizeof.cl_mem, Pointer.to(B2arrayMem));
		clSetKernelArg(_openCLManager._samplePopulationKernel, 4, Sizeof.cl_mem, Pointer.to(arx2arrayMem));
		clSetKernelArg(_openCLManager._samplePopulationKernel, 5, Sizeof.cl_mem, Pointer.to(xmeanMem));
		clSetKernelArg(_openCLManager._samplePopulationKernel, 6, Sizeof.cl_mem, Pointer.to(randomArrayMem));
		clSetKernelArg(_openCLManager._samplePopulationKernel, 7, Sizeof.cl_float, Pointer.to(new float[]{ sigma }));
		
		//Kernel - storeBest
		clSetKernelArg(_openCLManager._storeBestKernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ populationSize }));
		clSetKernelArg(_openCLManager._storeBestKernel, 1, Sizeof.cl_int, Pointer.to(new int[]{ N }));
		clSetKernelArg(_openCLManager._storeBestKernel, 2, Sizeof.cl_mem, Pointer.to(arx2arrayMem));
		clSetKernelArg(_openCLManager._storeBestKernel, 3, Sizeof.cl_mem, null);
		clSetKernelArg(_openCLManager._storeBestKernel, 4, Sizeof.cl_mem, Pointer.to(sumEvalMem));
		clSetKernelArg(_openCLManager._storeBestKernel, 5, Sizeof.cl_mem, Pointer.to(bestSolutionEverArrayMem));
		clSetKernelArg(_openCLManager._storeBestKernel, 6, Sizeof.cl_mem, Pointer.to(arfitnessMem));
		
		//Kernel - computeFitnessKernel
		clSetKernelArg(_openCLManager._computeFitnessKernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ populationSize }));
		clSetKernelArg(_openCLManager._computeFitnessKernel, 1, Sizeof.cl_int, Pointer.to(new int[]{ N }));
		clSetKernelArg(_openCLManager._computeFitnessKernel, 2, Sizeof.cl_mem, Pointer.to(arfitnessMem));
		clSetKernelArg(_openCLManager._computeFitnessKernel, 3, Sizeof.cl_mem, Pointer.to(arindexMem));
		
		//Kernel - calculateXmeanKernel
		clSetKernelArg(_openCLManager._calculateXmeanKernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ populationSize }));
		clSetKernelArg(_openCLManager._calculateXmeanKernel, 1, Sizeof.cl_int, Pointer.to(new int[]{ N }));
		clSetKernelArg(_openCLManager._calculateXmeanKernel, 2, Sizeof.cl_mem, Pointer.to(arindexMem));
		clSetKernelArg(_openCLManager._calculateXmeanKernel, 3, Sizeof.cl_mem, Pointer.to(arx2arrayMem));
		clSetKernelArg(_openCLManager._calculateXmeanKernel, 4, Sizeof.cl_mem, Pointer.to(weightsMem));
		clSetKernelArg(_openCLManager._calculateXmeanKernel, 5, Sizeof.cl_mem, Pointer.to(xmeanMem));
		clSetKernelArg(_openCLManager._calculateXmeanKernel, 6, Sizeof.cl_mem, Pointer.to(xoldMem));
		
		//Kernel - updateEvolutionPathsKernel
		clSetKernelArg(_openCLManager._updateEvolutionPathsKernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ N }));
		clSetKernelArg(_openCLManager._updateEvolutionPathsKernel, 1, Sizeof.cl_mem, Pointer.to(artmpMem));
		clSetKernelArg(_openCLManager._updateEvolutionPathsKernel, 2, Sizeof.cl_mem, Pointer.to(invsqrtC2arrayMem));
		clSetKernelArg(_openCLManager._updateEvolutionPathsKernel, 3, Sizeof.cl_mem, Pointer.to(xmeanMem));
		clSetKernelArg(_openCLManager._updateEvolutionPathsKernel, 4, Sizeof.cl_mem, Pointer.to(xoldMem));
		clSetKernelArg(_openCLManager._updateEvolutionPathsKernel, 5, Sizeof.cl_mem, Pointer.to(psMem));
		clSetKernelArg(_openCLManager._updateEvolutionPathsKernel, 6, Sizeof.cl_float, Pointer.to(new float[]{ sigma }));
		clSetKernelArg(_openCLManager._updateEvolutionPathsKernel, 7, Sizeof.cl_float, Pointer.to(new float[]{ cs }));
		clSetKernelArg(_openCLManager._updateEvolutionPathsKernel, 8, Sizeof.cl_float, Pointer.to(new float[]{ mueff }));
		
		//Kernel - calculateNormKernel
		clSetKernelArg(_openCLManager._calculateNormKernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ N }));
		clSetKernelArg(_openCLManager._calculateNormKernel, 1, Sizeof.cl_mem, Pointer.to(psMem));
		clSetKernelArg(_openCLManager._calculateNormKernel, 2, Sizeof.cl_mem, null);
		clSetKernelArg(_openCLManager._calculateNormKernel, 3, Sizeof.cl_mem, Pointer.to(psxpsMem));
		
		//Kernel - adaptCovarianceMatrixKernel
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ populationSize }));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 1, Sizeof.cl_int, Pointer.to(new int[]{ N }));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{ counteval }));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 3, Sizeof.cl_float, Pointer.to(new float[]{ cmu }));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 4, Sizeof.cl_float, Pointer.to(new float[]{ c1 }));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 5, Sizeof.cl_float, Pointer.to(new float[]{ cc }));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 6, Sizeof.cl_float, Pointer.to(new float[]{ sigma }));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 7, Sizeof.cl_float, Pointer.to(new float[]{ cs }));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 8, Sizeof.cl_float, Pointer.to(new float[]{ chiN }));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 9, Sizeof.cl_float, Pointer.to(new float[]{ mueff }));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 10, Sizeof.cl_mem, Pointer.to(psxpsMem));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 11, Sizeof.cl_mem, Pointer.to(xmeanMem));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 12, Sizeof.cl_mem, Pointer.to(xoldMem));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 13, Sizeof.cl_mem, Pointer.to(C2arrayMem));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 14, Sizeof.cl_mem, Pointer.to(pcMem));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 15, Sizeof.cl_mem, Pointer.to(weightsMem));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 16, Sizeof.cl_mem, Pointer.to(arindexMem));
		clSetKernelArg(_openCLManager._adaptCovarianceMatrixKernel, 17, Sizeof.cl_mem, Pointer.to(arx2arrayMem));
		
		//Kernel - updateDistributionHelperKernel
		clSetKernelArg(_openCLManager._updateDistributionHelperKernel, 0, Sizeof.cl_int, Pointer.to(new int[]{ N }));
		clSetKernelArg(_openCLManager._updateDistributionHelperKernel, 1, Sizeof.cl_mem, Pointer.to(artmp22arrayMem));
		clSetKernelArg(_openCLManager._updateDistributionHelperKernel, 2, Sizeof.cl_mem, Pointer.to(B2arrayMem));
		clSetKernelArg(_openCLManager._updateDistributionHelperKernel, 3, Sizeof.cl_mem, Pointer.to(diagDMem));
		clSetKernelArg(_openCLManager._updateDistributionHelperKernel, 4, Sizeof.cl_mem, Pointer.to(invsqrtC2arrayMem));
		clSetKernelArg(_openCLManager._updateDistributionHelperKernel, 5, Sizeof.cl_mem, Pointer.to(C2arrayMem));
		clSetKernelArg(_openCLManager._updateDistributionHelperKernel, 6, Sizeof.cl_mem, Pointer.to(offdiagMem));
	}
	
	//########################### CMA-ES helper methods ###########################//
	
	/**
	 * Method - getSolutionFromArray
	 * @param x
	 * @return
	 * @throws JMException
	 * @throws ClassNotFoundException
	 */
	private Solution getSolutionFromArray(float[] x) throws JMException, ClassNotFoundException{
		Solution solution = new Solution(problem_);
		for (int i = 0; i < N; i++) {
			solution.getDecisionVariables()[i].setValue(x[i]);
		}
		//Set the evaluation variable
		solution.setObjective(0, x[N]);
		return solution;
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
}
