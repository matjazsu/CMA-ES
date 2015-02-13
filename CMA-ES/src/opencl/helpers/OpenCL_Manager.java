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
    public final long cl_deviceType = CL_DEVICE_TYPE_ALL;
    public final int cl_deviceIndex = 0;
    public static cl_context context;
    public static cl_command_queue commandQueue;
    public static cl_kernel kernel;
    public static cl_program program;
    public static cl_device_id[] devices;
    public static cl_device_id device;
    public int cl_numDevices;
    public int cl_numberOfComputeUnits; //Stores the selected device number of Compute Units
    public long cl_workGroupSize; //Stores the selected decice Work Group size
    
    public Logger cl_logger; //Logger object
	public FileHandler cl_fileHandler; //FileHandler object
	
	//Constructor
	public OpenCL_Manager() throws JMException, IOException{
		//Initialize logger and file handler
		cl_logger = Logger.getLogger("OpenCL_Manager");
		cl_fileHandler = new FileHandler("OpenCL_Manager.log");
		cl_logger.addHandler(cl_fileHandler);
		
		//Initialize OpenCL device
		InitOpenCLDevice();
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
	        device = selectTheBestDevice(devices);
	        
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
			}
			catch(Exception ex){
				throw ex;
			}
		}
	}
	
	public void ReleaseOpenCLEnvironment(){
		try{
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
