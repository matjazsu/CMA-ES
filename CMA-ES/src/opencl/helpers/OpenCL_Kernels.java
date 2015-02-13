package opencl.helpers;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class OpenCL_Kernels {
	
	/**
	 * Returns kernel as String
	 * @param kernelName
	 * @return
	 * @throws IOException 
	 */
	public static String GetKernel(String kernelName) throws IOException{
		
		//Initialize InputStream
		InputStream is = OpenCL_Kernels.class.getResourceAsStream(kernelName);
		
		//Initialize BufferedReader 
		BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        StringBuilder kernel = new StringBuilder();
        String line;
        
        //Read until the end
        while ((line = reader.readLine()) != null) {
            kernel.append(line + "\n");
        }
        
        //Close BufferedReader
		reader.close();
		
        return kernel.toString();
	}

}
