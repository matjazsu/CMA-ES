package opencl.helpers;

import static org.jocl.CL.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

import jmetal.util.JMException;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

/**
 * This class is used for managing OpenCL environment
 * @author matjaz
 * @date 02.2015
 */
public class OpenCL_Manager {

	//Public properties
    public final int cl_platformIndex = 0;
    private static cl_platform_id platform;
    public final long cl_deviceType = CL_DEVICE_TYPE_GPU;
    public final int cl_deviceIndex = 0;
    private int cl_selectedDevice;
    
    //context
    public static cl_context context;
    //commandQueue
    public static cl_command_queue commandQueue;
    
    //samplePopulation
    public static cl_kernel _samplePopulationKernel;
    public static cl_program _samplePopulationProgram;
    
    //storeBest
    public static cl_kernel _storeBestKernel;
    public static cl_program _storeBestProgram;
    
    //computeFitness
    public static cl_kernel _computeFitnessKernel;
    public static cl_program _computeFitnessProgram;
    
    //calculateXmean
    public static cl_kernel _calculateXmeanKernel;
    public static cl_program _calculateXmeanProgram;
    
    //updateEvolutionPaths
    public static cl_kernel _updateEvolutionPathsKernel;
    public static cl_program _updateEvolutionPathsProgram;
    
    //calculateNorm
    public static cl_kernel _calculateNormKernel;
    public static cl_program _calculateNormProgram;
    
    //adaptCovarianceMatrix
    public static cl_kernel _adaptCovarianceMatrixKernel;
    public static cl_program _adaptCovarianceMatrixProgram;
    
    //updateDistributionHelper
    public static cl_kernel _updateDistributionHelperKernel;
    public static cl_program _updateDistributionHelperProgram;
    
    //device
    public static cl_device_id[] devices;
    public static cl_device_id device;
    public int cl_numDevices;
    
    //device info
    public int cl_numberOfComputeUnits; //Stores the selected device number of Compute Units
    public long cl_workGroupSize; //Stores the selected decice Work Group size
    
    //logger
    public Logger cl_logger; //Logger object
	public FileHandler cl_fileHandler; //FileHandler object
	
	//OpenCL Kernels variables
	private String samplePopulationKernelSource = "";
	private String storeBestKernelSource = "";
	private String computeFitnessKernelSource = "";
	private String calculateXmeanKernelSource = "";
	private String updateEvolutionPathsKernelSource = "";
	private String calculateNormKernelSource = "";
	private String adaptCovarianceMatrixKernelSource = "";
	private String updateDistributionHelperKernelSource = "";
	
	//Constructor
	public OpenCL_Manager(int selectedDevice) throws Exception{
		//Initialize logger and file handler
		cl_logger = Logger.getLogger("OpenCL_Manager");
		cl_fileHandler = new FileHandler("OpenCL_Manager.log");
		cl_logger.addHandler(cl_fileHandler);
		
		//Initialize properties
		this.cl_selectedDevice = selectedDevice;
		
		//Initialize OpenCL device
		InitOpenCLDevice();
		
		//Init OpenCL kernels
		initOpenCLKernels();
	}
	
	/**
	 * Finds all capable OpenCL devices and selects the best one  
	 * @throws Exception
	 */
	public void InitOpenCLDevice() throws JMException{
		
		try{
			
			//Enable exceptions and subsequently omit error checks
	        CL.setExceptionsEnabled(true);
	        
	        //############################# Platform selection #############################//
	        
	        //Obtain the number of platforms
	        int numPlatformsArray[] = new int[1];
	        clGetPlatformIDs(0, 
	        		null, 
	        		numPlatformsArray);
	        int numPlatforms = numPlatformsArray[0];
	        
	        //Obtain a platform ID
	        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
	        clGetPlatformIDs(platforms.length, 
	        		platforms, 
	        		null);
	        platform = platforms[cl_platformIndex];
	        String platformName = getString(platform, CL_PLATFORM_NAME);
	        this.cl_logger.info("Using platform " + cl_platformIndex + " of " + numPlatforms + ": " + platformName);
		    
	        //############################# Devices selection #############################//
	        
	        //Obtain the number of devices for the platform
	        int numDevicesArray[] = new int[1];
	        clGetDeviceIDs(platform, 
	        		cl_deviceType, 
	        		0, 
	        		null, 
	        		numDevicesArray);
	        cl_numDevices = numDevicesArray[0];
	        
	        //Obtain a device ID 
	        devices = new cl_device_id[cl_numDevices];
	        clGetDeviceIDs(platform, 
	        		cl_deviceType, 
	        		cl_numDevices, 
	        		devices, 
	        		null);
	        for (int i = 0; i < cl_numDevices; i++)
	        {
	        	//Get device basic information
	            String deviceName = getString(devices[i], CL_DEVICE_NAME);
	            //Get max number of Compute Units
				int maxComputeUnits = getInt(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS);
				//Get max Work Group size
				long maxWorkGroupSize = getSize(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE);
	            
				String informations = "";
				informations += "\n --- Device " + i + " of " + cl_numDevices + ": \n";
				informations += "CL_DEVICE_NAME: " + deviceName + "\n";
				informations += "CL_DEVICE_MAX_COMPUTE_UNITS: " + maxComputeUnits + "\n";
				informations += "CL_DEVICE_MAX_WORK_GROUP_SIZE: " + maxWorkGroupSize + "\n";
				
				//Log informations to file
	            this.cl_logger.info(informations);
	        }
	        
	        //Select device with the best performance
	        if(this.cl_selectedDevice >= 0){
	        	device = selectDeviceWithIndex(this.cl_selectedDevice);
	        }
	        else{
	        	device = selectTheBestDevice(devices);
	        }
	        
	        //Log device information
	        printDeviceInfo(device);
		}
		catch(Exception ex){
			throw ex;
		}
	}
	
	/**
	 * This method initialized OpenCL context and command queue
	 */
	public void InitOpenCLContext() throws JMException{
		
		if(context == null){
			try{
		        
				//############################# Context initialization #############################//
		        
		        //Initialize the context properties
		        cl_context_properties contextProperties = new cl_context_properties();
		        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
				
		        //############################# Context creation #############################//
	
		        //Create a context for the selected device
		        context = clCreateContext(
		            contextProperties, 
		            1, 
		            new cl_device_id[]{device}, 
		            null, 
		            null, 
		            null);
		        
		        //############################# CommandQueue creation #############################//
		        
		        //Create a command-queue for the selected device
		        commandQueue = clCreateCommandQueue(context, device, 0, null);
		        
		        //############################# Create/Build samplePopulation program #############################//
		        
		        createSamplePopulationProgram();
		        
		        //############################# Create/Build storeBest program #############################//
		        
		        storeBestProgram();
		        
		        //############################# Create/Build computeFitness program #############################//
		        
		        computeFitnessProgram();
		        
		        //############################# Create/Build calculateXmean program #############################//
		        
		        calculateXmeanProgram();
		        
		        //############################# Create/Build updateEvolutionPaths program #############################//
		        
		        updateEvolutionPathsProgram();
		        
		        //############################# Create/Build calculateNorm program #############################//
		        
		        calculateNormProgram();
		        
		        //############################# Create/Build adaptCovarianceMatrix program #############################//
		        
		        adaptCovarianceMatrixProgram();
		        
		        //############################# Create/Build updateDistributionHelper program #############################//
		        
		        createUpdateDistributionHelperProgram();
			}
			catch(Exception ex){
				throw ex;
			}
		}
	}
	
	public void ReleaseOpenCLEnvironment(){
		try{

			//Release samplePopulation kernel
			clReleaseKernel(_samplePopulationKernel);
			//Release samplePopulation program
			clReleaseProgram(_samplePopulationProgram);
			
			//Release storeBest kernel
			clReleaseKernel(_storeBestKernel);
			//Release storeBest program
			clReleaseProgram(_storeBestProgram);
			
			//Release computeFitness kernel
			clReleaseKernel(_computeFitnessKernel);
			//Release computeFitness program
			clReleaseProgram(_computeFitnessProgram);
			
			//Release computeFitness kernel
			clReleaseKernel(_calculateXmeanKernel);
			//Release computeFitness program
			clReleaseProgram(_calculateXmeanProgram);
			
			//Release computeFitness kernel
			clReleaseKernel(_updateEvolutionPathsKernel);
			//Release computeFitness program
			clReleaseProgram(_updateEvolutionPathsProgram);
			
			//Release updateDistribution kernel
			clReleaseKernel(_calculateNormKernel);
			//Release updateDistribution program
			clReleaseProgram(_calculateNormProgram);
			
			//Release adaptCovarianceMatrix kernel
			clReleaseKernel(_adaptCovarianceMatrixKernel);
			//Release adaptCovarianceMatrix program
			clReleaseProgram(_adaptCovarianceMatrixProgram);
			
			//Release updateDistributionHelper kernel
			clReleaseKernel(_updateDistributionHelperKernel);
			//Release updateDistributionHelper program
			clReleaseProgram(_updateDistributionHelperProgram);
			
			//Release CommandQueue
			clReleaseCommandQueue(commandQueue);
			
			//Release Context
	        clReleaseContext(context);
		}
		catch(Exception ex){
			throw ex;
		}
	}
	
	/////---------------------------------------/////
	
	/*
	 * Helper methods
	 */
	
	/////---------------------------------------/////
	
	/**
	 * Read and set OpenCL Kernels
	 * @throws Exception 
	 */
	private void initOpenCLKernels() throws Exception{
		//Read samplePopulationKernel
		samplePopulationKernelSource = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_samplePopulation);
		
		//Read storeBestKernel
		storeBestKernelSource = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_storeBest);
		
		//Read computeFitnessKernel
		computeFitnessKernelSource = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_computeFitness);
		
		//Read calculateXmeanKernel
		calculateXmeanKernelSource = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_calculateXmean);
		
		//Read updateEvolutionPathsKernel
		updateEvolutionPathsKernelSource = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_updateEvolutionPaths);
		
		//Read calculateNormKernel
		calculateNormKernelSource = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_calculateNorm);
		
		//Read adaptCovarianceMatrixKernel
		adaptCovarianceMatrixKernelSource = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_adaptCovarianceMatrix);

		//Read updateDistributionHelperKernel 
		updateDistributionHelperKernelSource = OpenCL_Kernels.GetKernel(OpenCL_Kernels_Enums.path_updateDistributionHelper);
	}
	
	/**
	 * Creates OpenCL program for updateDistribution kernel
	 */
	private void createSamplePopulationProgram(){
		
		//Create the program from the source code
		_samplePopulationProgram = clCreateProgramWithSource(context,
				1, 
				new String[]{ samplePopulationKernelSource }, 
				null, 
				null);

		//Build the program
		clBuildProgram(_samplePopulationProgram, 
				0, 
				null, 
				null, 
				null, 
				null);

		//Create the kernel
		_samplePopulationKernel = clCreateKernel(_samplePopulationProgram, 
				OpenCL_Kernels_Enums.name_samplePopulation, 
				null);
	}
	
	/**
	 * Creates OpenCL program for updateDistribution kernel
	 */
	private void storeBestProgram(){
		
		//Create the program from the source code
		_storeBestProgram = clCreateProgramWithSource(context,
				1, 
				new String[]{ storeBestKernelSource }, 
				null, 
				null);

		//Build the program
		clBuildProgram(_storeBestProgram, 
				0, 
				null, 
				null, 
				null, 
				null);

		//Create the kernel
		_storeBestKernel = clCreateKernel(_storeBestProgram, 
				OpenCL_Kernels_Enums.name_storeBest, 
				null);
	}
	
	/**
	 * Creates OpenCL program for computeFitness kernel
	 */
	private void computeFitnessProgram(){
		
		//Create the program from the source code
		_computeFitnessProgram = clCreateProgramWithSource(context,
				1, 
				new String[]{ computeFitnessKernelSource }, 
				null, 
				null);

		//Build the program
		clBuildProgram(_computeFitnessProgram, 
				0, 
				null, 
				null, 
				null, 
				null);

		//Create the kernel
		_computeFitnessKernel = clCreateKernel(_computeFitnessProgram, 
				OpenCL_Kernels_Enums.name_computeFitness, 
				null);
	}
	
	/**
	 * Creates OpenCL program for computeFitness kernel
	 */
	private void calculateXmeanProgram(){
		
		//Create the program from the source code
		_calculateXmeanProgram = clCreateProgramWithSource(context,
				1, 
				new String[]{ calculateXmeanKernelSource }, 
				null, 
				null);

		//Build the program
		clBuildProgram(_calculateXmeanProgram, 
				0, 
				null, 
				null, 
				null, 
				null);

		//Create the kernel
		_calculateXmeanKernel = clCreateKernel(_calculateXmeanProgram, 
				OpenCL_Kernels_Enums.name_calculateXmean, 
				null);
	}
	
	/**
	 * Creates OpenCL program for updateEvolutionPaths kernel
	 */
	private void updateEvolutionPathsProgram(){
		
		//Create the program from the source code
		_updateEvolutionPathsProgram = clCreateProgramWithSource(context,
				1, 
				new String[]{ updateEvolutionPathsKernelSource }, 
				null, 
				null);

		//Build the program
		clBuildProgram(_updateEvolutionPathsProgram, 
				0, 
				null, 
				null, 
				null, 
				null);

		//Create the kernel
		_updateEvolutionPathsKernel = clCreateKernel(_updateEvolutionPathsProgram, 
				OpenCL_Kernels_Enums.name_updateEvolutionPaths, 
				null);
	}
	
	/**
	 * Creates OpenCL program for calculateNorm kernel
	 */
	private void calculateNormProgram(){
		
		//Create the program from the source code
		_calculateNormProgram = clCreateProgramWithSource(context,
				1, 
				new String[]{ calculateNormKernelSource }, 
				null, 
				null);

		//Build the program
		clBuildProgram(_calculateNormProgram, 
				0, 
				null, 
				null, 
				null, 
				null);

		//Create the kernel
		_calculateNormKernel = clCreateKernel(_calculateNormProgram, 
				OpenCL_Kernels_Enums.name_calculateNorm, 
				null);
	}
	
	/**
	 * Creates OpenCL program for adaptCovarianceMatrix kernel
	 */
	private void adaptCovarianceMatrixProgram(){
		
		//Create the program from the source code
		_adaptCovarianceMatrixProgram = clCreateProgramWithSource(context,
				1, 
				new String[]{ adaptCovarianceMatrixKernelSource }, 
				null, 
				null);

		//Build the program
		clBuildProgram(_adaptCovarianceMatrixProgram, 
				0, 
				null, 
				null, 
				null, 
				null);

		//Create the kernel
		_adaptCovarianceMatrixKernel = clCreateKernel(_adaptCovarianceMatrixProgram, 
				OpenCL_Kernels_Enums.name_adaptCovarianceMatrix, 
				null);
	}
	
	/**
	 * Creates OpenCL program for updateDistribution kernel
	 */
	private void createUpdateDistributionHelperProgram(){
		
		//Create the program from the source code
		_updateDistributionHelperProgram = clCreateProgramWithSource(context,
				1, 
				new String[]{ updateDistributionHelperKernelSource }, 
				null, 
				null);

		//Build the program
		clBuildProgram(_updateDistributionHelperProgram, 
				0, 
				null, 
				null, 
				null, 
				null);

		//Create the kernel
		_updateDistributionHelperKernel = clCreateKernel(_updateDistributionHelperProgram, 
				OpenCL_Kernels_Enums.name_updateDistributionHelper, 
				null);
	}
	
	/**
	 * Returns device with the best performance
	 * 
	 * @param devices
	 * @return cl_device_id
	 */
	private cl_device_id selectTheBestDevice(cl_device_id[] devices) {
		
		//Initialize temporary properties
		long bestPerformace = 0;
		
		//Initially set bestDevice to the first found device
		cl_device_id bestDevice = devices[0];
		//Initially set the number of Compute Units
		cl_numberOfComputeUnits = getInt(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS);
		//Initially set the Work Group size
		cl_workGroupSize = getSize(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE);
		
		for (cl_device_id device : devices){
			
			//Get max number of Compute Units
			int maxComputeUnits = getInt(device, CL_DEVICE_MAX_COMPUTE_UNITS);
			//Get max Work Group size
			long maxWorkGroupSize = getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
			
			//If the current device has better performance --> Select it
			if(bestPerformace < (maxComputeUnits * maxWorkGroupSize)){
				//Update best device
				bestDevice = device;
				//Update device Compute units
				cl_numberOfComputeUnits = maxComputeUnits;
				//Update device work group size
				cl_workGroupSize = maxWorkGroupSize;
				bestPerformace = (maxComputeUnits * maxWorkGroupSize);
			}
		}
		
		//Return device with the best performance 
		return bestDevice;
	}
	
	/**
	 * Returns device with the given index
	 * @param deviceIndex
	 * @return
	 */
	private cl_device_id selectDeviceWithIndex(int deviceIndex){
		
		//Select device with the given index
		cl_device_id selectedDevice = devices[deviceIndex];
		//Set the number of Compute Units
		cl_numberOfComputeUnits = getInt(devices[deviceIndex], CL_DEVICE_MAX_COMPUTE_UNITS);
		//Set the Work Group size
		cl_workGroupSize = getSize(devices[deviceIndex], CL_DEVICE_MAX_WORK_GROUP_SIZE);
		
		//Return selected device 
		return selectedDevice;
	}
	
	/**
	 * Log information about the given device 
	 * 
	 * @param devices
	 */
	public void printDeviceInfo(cl_device_id device) {
		
		String information = "";
		
    	// CL_DEVICE_NAME
        String deviceName = getString(device, CL_DEVICE_NAME);
        information += "\n--- Info for selected device " + deviceName + ": ---\n";
        information += "CL_DEVICE_NAME: " + deviceName + "\n";

        // CL_DEVICE_VENDOR
        String deviceVendor = getString(device, CL_DEVICE_VENDOR);
        information += "CL_DEVICE_VENDOR: " + deviceVendor + "\n";

        // CL_DRIVER_VERSION
        String driverVersion = getString(device, CL_DRIVER_VERSION);
        information += "CL_DRIVER_VERSION: " + driverVersion + "\n";
        
        // CL_DEVICE_MAX_COMPUTE_UNITS
        int maxComputeUnits = getInt(device, CL_DEVICE_MAX_COMPUTE_UNITS);
        information += "CL_DEVICE_MAX_COMPUTE_UNITS: " + maxComputeUnits + "\n";
        
        // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
        long maxWorkItemDimensions = getLong(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
        information += "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " + maxWorkItemDimensions + "\n";
        
        // CL_DEVICE_MAX_WORK_ITEM_SIZES
        long maxWorkItemSizes[] = getSizes(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
        information += "CL_DEVICE_MAX_WORK_ITEM_SIZES: " + maxWorkItemSizes[0] + "/" + 
        						maxWorkItemSizes[1] + "/" +
        						maxWorkItemSizes[2] + "\n";

        // CL_DEVICE_MAX_WORK_GROUP_SIZE
        long maxWorkGroupSize = getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        information += "CL_DEVICE_MAX_WORK_GROUP_SIZE: " + maxWorkGroupSize + "\n";
        
        //CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE
        int doublePrecisionFirstCheck = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
        information += "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: " + doublePrecisionFirstCheck + "\n";
        
        //CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE
        int doublePrecisionSecondCheck = getInt(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE);
        information += "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE: " + doublePrecisionSecondCheck + "\n";
        
        //Save to log file
        this.cl_logger.info(information);
	}
	
	/**
	 * Return the local work size 
	 * OpenCL 
	 * @return
	 */
	public long getLocalWorkSize(int globalSize){
		return globalSize >= this.cl_workGroupSize ? this.cl_workGroupSize : globalSize;
	}
	
	/**
	 * Returns the global work size
	 * @param groupSize
	 * @param globalSize
	 * @return
	 */
	public long getGlobalWorkSize(long groupSize, int globalSize){
		long r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
	}
	
	/**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private int getInt(cl_device_id device, int paramName)
    {
        return getInts(device, paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    private int[] getInts(cl_device_id device, int paramName, int numValues)
    {
        int values[] = new int[numValues];
        clGetDeviceInfo(device, paramName, Sizeof.cl_int * numValues, Pointer.to(values), null);
        return values;
    }
    
    /**
     * Returns the value of the platform info parameter with the given name
     *
     * @param platform The platform
     * @param paramName The parameter name
     * @return The value
     */
    private static String getString(cl_platform_id platform, int paramName)
    {
        long size[] = new long[1];
        clGetPlatformInfo(platform, paramName, 0, null, size);
        byte buffer[] = new byte[(int)size[0]];
        clGetPlatformInfo(platform, paramName, 
            buffer.length, Pointer.to(buffer), null);
        return new String(buffer, 0, buffer.length-1);
    }
	
	/**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private String getString(cl_device_id device, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private long getLong(cl_device_id device, int paramName)
    {
        return getLongs(device, paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    private long[] getLongs(cl_device_id device, int paramName, int numValues)
    {
        long values[] = new long[numValues];
        clGetDeviceInfo(device, paramName, Sizeof.cl_long * numValues, Pointer.to(values), null);
        return values;
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private long getSize(cl_device_id device, int paramName)
    {
        return getSizes(device, paramName, 1)[0];
    }
    
    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    private long[] getSizes(cl_device_id device, int paramName, int numValues)
    {
        // The size of the returned data has to depend on 
        // the size of a size_t, which is handled here
        ByteBuffer buffer = ByteBuffer.allocate(
            numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
        clGetDeviceInfo(device, paramName, Sizeof.size_t * numValues, 
            Pointer.to(buffer), null);
        long values[] = new long[numValues];
        if (Sizeof.size_t == 4)
        {
            for (int i=0; i<numValues; i++)
            {
                values[i] = buffer.getInt(i * Sizeof.size_t);
            }
        }
        else
        {
            for (int i=0; i<numValues; i++)
            {
                values[i] = buffer.getLong(i * Sizeof.size_t);
            }
        }
        return values;
    }
	
}
