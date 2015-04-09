package opencl.helpers;

public class OpenCL_Kernels_Enums {
	
	//OpenCL kernel - samplePopulation 
	public static final String name_samplePopulation = "samplePopulation";
	public static final String path_samplePopulation = "samplePopulationKernel.cl";
	
	//OpenCL kernel - samplePopulation 
	public static final String name_storeBest = "storeBest";
	public static final String path_storeBest = "storeBestKernel.cl";
	
	//OpenCL kernel - computeFitness 
	public static final String name_computeFitness = "computeFitness";
	public static final String path_computeFitness = "computeFitnessKernel.cl";
	
	//OpenCL kernel - calculateXmean 
	public static final String name_calculateXmean = "calculateXmean";
	public static final String path_calculateXmean = "calculateXmeanKernel.cl";
	
	//OpenCL kernel - updateEvolutionPaths 
	public static final String name_updateEvolutionPaths = "updateEvolutionPaths";
	public static final String path_updateEvolutionPaths = "updateEvolutionPathsKernel.cl";
	
	//OpenCL kernel - calculateNorm 
	public static final String name_calculateNorm = "calculateNorm";
	public static final String path_calculateNorm = "calculateNormKernel.cl";
	
	//OpenCL kernel - adaptCovarianceMatrix 
	public static final String name_adaptCovarianceMatrix = "adaptCovarianceMatrix";
	public static final String path_adaptCovarianceMatrix = "adaptCovarianceMatrixKernel.cl";
	
	//OpenCL kernel - updateDistributionHelper 
	public static final String name_updateDistributionHelper = "updateDistributionHelper";
	public static final String path_updateDistributionHelper = "updateDistributionHelperKernel.cl";
}
