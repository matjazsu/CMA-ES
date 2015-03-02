package cmaes.main;

import opencl.metaheuristics.singleObjective.cmaes.OpenCL_CMAES_main;
import jmetal.core.Problem;
import jmetal.metaheuristics.singleObjective.cmaes.CMAES_main;
import jmetal.problems.singleObjective.Griewank;
import jmetal.problems.singleObjective.Rastrigin;
import jmetal.problems.singleObjective.Rosenbrock;
import jmetal.problems.singleObjective.Sphere;

/**
 * This class is the main class of the CMA-ES project
 * Initializes either OpenCL or Iterative implementation of CMA-ES algorithm
 * @author matjaz suber
 * @date 02.2015
 */
public class CMAES_start {
	
	//Private properties
	private static int cmaes_version;
	private static String cmaes_problem;
	private static int cmaes_nrVariables;
	private static int cmaes_populationSize;
	private static int cmaes_maxEvaluations;
	
	/**
	 * Method main
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		//Problem default initialization
		Problem problem = null;
		
		try{
			
			//##################### Read input parameters #####################// 
			
			//Get version
			//Version == 0 --> Iterative implementation
			//Version == 1 --> OpenCL implementation
			cmaes_version = Integer.parseInt(args[0]);
			//cmaes_version = 1;
			
			//Get problem
			//Problems: Griewank, Sphere, Rosenbrock, Rastrigin
			cmaes_problem = args[1];
			//cmaes_problem = "Rosenbrock";
			
			//Get number of problem variables
			cmaes_nrVariables = Integer.parseInt(args[2]);
			//cmaes_nrVariables = 128;
			
			//Get population size			
			cmaes_populationSize = Integer.parseInt(args[3]);
			//cmaes_populationSize = 128;
			
			//Get maxEvaluations
			cmaes_maxEvaluations = Integer.parseInt(args[4]);
			//cmaes_maxEvaluations = 264;			
			
			//##################### Set/Initialize problem #####################//
			
			switch(cmaes_problem){
			
				case CMAES_start_problems.cmaes_Griewank:
					problem = new Griewank("Real", cmaes_nrVariables);
					break;
					
				case CMAES_start_problems.cmaes_Sphere:
					problem = new Sphere("Real", cmaes_nrVariables);
					break;
				
				case CMAES_start_problems.cmaes_Rosenbrock:
					problem = new Rosenbrock("Real", cmaes_nrVariables);
					break;
					
				case CMAES_start_problems.cmaes_Rastrigin:
					problem = new Rastrigin("Real", cmaes_nrVariables);
					break;
					
				default:
					problem = null;
					break;
			}
		}
		catch(Exception ex){
			//Print error message to console
			System.out.println("Please enter all required parameters.\n" +
							   "Example: java -jar CMA-ES.jar 0 Rosenbrock 10 10 1000");
			System.exit(0);
		}

		if(cmaes_version >= 0){
			if(problem != null){
				
				if(cmaes_version == 0){
					//Iterative version
					CMAES_main cmaesMain;
					try {
						cmaesMain = new CMAES_main(cmaes_populationSize, 
												   cmaes_maxEvaluations,
												   problem);
						//Execute
						cmaesMain.Execute();
					} catch (Exception ex) {
						// TODO Auto-generated catch block
						System.out.println("ERROR in Iterative version. See the log file.");
						System.exit(0);
					}
				}
				else if(cmaes_version == 1){
					//OpenCL version
					OpenCL_CMAES_main clcmaesMain;
					try {
						clcmaesMain = new OpenCL_CMAES_main(cmaes_populationSize, 
															cmaes_maxEvaluations,
															problem);
						//Execute
						clcmaesMain.Execute();
					} catch (Exception ex) {
						// TODO Auto-generated catch block
						System.out.println("ERROR in OpenCL version. See the log file.");
						System.exit(0);
					}
				}
				else{
					//Print error message to console
					System.out.println("Please enter a valid version.\n" + 
									   "0 - Iterative implementation,\n" +
									   "1 - OpenCL implementation.");
					System.exit(0);
				}
				
			}
			else{
				//Print error message to console
				System.out.println("Please enter one of the folowing problems.\n" +
								   "Problems: Griewank, Sphere, Rosenbrock, Rastrigin.");
				System.exit(0);
			}
		}
		else{
			//Print error message to console
			System.out.println("Please enter a valid version.\n" + 
							   "0 - Iterative implementation,\n" +
							   "1 - OpenCL implementation.");
			System.exit(0);
		}
		
	}

}
