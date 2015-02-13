package opencl.metaheuristics.singleObjective.cmaes;

import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

import jmetal.core.Algorithm;
import jmetal.core.Problem;
import jmetal.core.SolutionSet;

/**
 * This class runs a single-objective CMA-ES algorithm.
 * OpenCL version (Host, CPU, GPU)
 * @author: matjaz suber
 * @date: 02.2015
 */
public class OpenCL_CMAES_main {
	
	//Private properties
	private int cl_populationSize;
	private int cl_maxEvaluations;
	private Problem cl_problem;
	
	//Public properties
	public Logger cl_logger; //Logger object
	public FileHandler cl_fileHandler; //FileHandler object
	
	//Constructor
	public OpenCL_CMAES_main(int populationSize, 
							 int maxEvaluations, 
							 Problem problem) throws SecurityException, IOException{
		
		//Initialize logger and file handler
		cl_logger = Logger.getLogger("OpenCL_CMAES_main");
		cl_fileHandler = new FileHandler("CMAES_OpenCL.log");
		cl_logger.addHandler(cl_fileHandler);
		
		//Initialite private properties
		this.cl_populationSize = populationSize;
		this.cl_maxEvaluations = maxEvaluations;
		this.cl_problem = problem;
		
		this.cl_logger.info("OpenCL version (Host, CPU, GPU) initialized successfully.");
	}
	
	/**
	 * Method Execute
	 */
	public void Execute() throws Exception{
		
		try{
			//######################## Declaration segment ########################//
		    
			//SolutionSet
		    SolutionSet population;
		    
		    //Algorithm
		    Algorithm algorithm;
		    
		    //######################## Initialization segment ########################//
		    
		    //Algorithm initialization - CMA ES on OpenCL
		    algorithm = new OpenCL_CMAES(this.cl_problem);
		    
		    //Setting algorithm parameters
		    algorithm.setInputParameter("populationSize", this.cl_populationSize);
		    algorithm.setInputParameter("maxEvaluations", this.cl_maxEvaluations);
		    
		    //######################## Execution segment ########################//
		    
		    //Algorithm execution with OpenCL
		    long initTime = System.currentTimeMillis();
		    population = algorithm.execute();
		    long estimatedTime = System.currentTimeMillis() - initTime;
		    
		    //######################## Log solutions to files ########################//
		    
		    this.cl_logger.info("Total execution time: " + estimatedTime + "ms");
		    this.cl_logger.info("Variables values have been writen to file VAR");
		    population.printVariablesToFile("VAR");    
		    this.cl_logger.info("Objectives values have been writen to file FUN");
		    population.printObjectivesToFile("FUN");
		}
		catch(Exception ex){
			ex.getStackTrace();
			this.cl_logger.severe("OpenCL_CMAES_main ERROR: " + ex.getMessage());
			throw ex;
		}
	}
}
