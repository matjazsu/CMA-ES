//  CMAES_main.java
//
//  Author:
//       Esteban López-Camacho <esteban@lcc.uma.es>
//
//  Copyright (c) 2013 Esteban López-Camacho
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
// 
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
package jmetal.metaheuristics.singleObjective.cmaes;

import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

import jmetal.core.Algorithm;
import jmetal.core.Problem;
import jmetal.core.SolutionSet;

/**
 * This class runs a single-objective CMA-ES algorithm.
 * Iterative version (CPU)
 * Mofidied by: matjaz suber
 * @date 02.2015
 */
public class CMAES_main {

	//Private properties
	private int cl_populationSize;
	private int cl_maxEvaluations;
	private Problem cl_problem;
	
	//Public properties
	public Logger cl_logger; //Logger object
	public FileHandler cl_fileHandler; //FileHandler object
	
	//Constructor
	public CMAES_main(int populationSize, 
				 	  int maxEvaluations, 
				 	  Problem problem) throws SecurityException, IOException{
	
		//Initialize logger and file handler
		cl_logger = Logger.getLogger("CMAES_main");
		cl_fileHandler = new FileHandler("CMAES_Iterative.log");
		cl_logger.addHandler(cl_fileHandler);
		
		//Initialite private properties
		this.cl_populationSize = populationSize;
		this.cl_maxEvaluations = maxEvaluations;
		this.cl_problem = problem;
		
		this.cl_logger.info("Iterative version (CPU) -- initialized successfully.");
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
		    algorithm = new CMAES(this.cl_problem);
		    
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
			this.cl_logger.severe("CMAES_main ERROR: " + ex.getMessage());
			throw ex;
		}
	}
}
