import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import geopandas as gpd
from rasterio.features import geometry_mask
import os
from osgeo import gdal


with rasterio.open('F:/myenv/deforestinput.tif') as deforest: 
         deforestRasterArray=deforest.read(1) # Read the deforest tif file
         profile=deforest.profile
   

def createRoadRaster():# function for creating road raster

       
    global deforestRasterArray
    shapeFilePath='F:/myenv/roadsdist/roadproj.shp'   
    roadGdf = gpd.read_file(shapeFilePath)# Read all the road shapefiles

    # Create a mask using the road geometries
    roadGeomGdf=gpd.GeoDataFrame(roadGdf,crs=roadGdf.crs)
    mask = geometry_mask(roadGeomGdf.geometry, out_shape=deforest.shape, transform=deforest.transform, invert=True)

    # Set pixels within the raster in the mask to 1
    deforestRasterArray[mask] = 1   
    deforestRasterArray[~mask]=0

    roadRasterPath = "F:/myenv/roadsdist" # road raster  output folder


    roadRaster= os.path.join(roadRasterPath, f"roadraster.tif")
    with rasterio.open(roadRaster, 'w', **profile) as dst:
        dst.write(deforestRasterArray, 1)
       





def computeDistance(input_file, dist_file, values=1,
                     nodata=4294967295, verbose=True): # function for computing distance from a given pixel value,obtained from Forest risk package
    
    """Computing the shortest distance to pixels with given values in
    a raster file.

    This function computes the shortest distance to pixels with given
    values in a raster file. Distances generated are in georeferenced
    coordinates.

    :param input_file: Input road raster file.

    :param dist_file: Path to the road distance raster file that is
        created.

    :param values: Values of the raster to compute the distance to. If
        several values, they must be separated with a comma in a
        string (eg. '0,1'). Default is 0.

    :param nodata: NoData value. Default is 4294967295 for UInt32.

    :param verbose: Logical. Whether to print messages or not. Default
        to ``True``.

    :return: None. A distance raster file is created (see
        ``dist_file``). Raster data type is UInt32 ([0,
        4294967295]).

    """

    # Read input file
    src_ds = gdal.Open(input_file)
    srcband = src_ds.GetRasterBand(1)

    # Create raster of distance
    drv = gdal.GetDriverByName("GTiff")
    dst_ds = drv.Create(
        dist_file,
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_UInt32,
        ["COMPRESS=LZW", "PREDICTOR=2", "BIGTIFF=YES"],
    )
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjectionRef())
    dstband = dst_ds.GetRasterBand(1)

    # Compute distance
    val = "VALUES=" + str(values)
    #cb =  gdal.TermProgress if verbose else None
    gdal.ComputeProximity(srcband, dstband, [val, "DISTUNITS=GEO"], callback=None)

    # Set nodata value
    dstband.SetNoDataValue(nodata)

    # Delete objects
    srcband = None
    dstband = None
    del src_ds, dst_ds



def processDeforestMap():# function for processing deforest forest raster
        
        global deforestRasterArray
        deforestRasterArray=np.where((np.logical_or(deforestRasterArray==1,deforestRasterArray==2)),1,0)# assign 1 for deforest classes and 0 elsewhere
        deforestRasterArray=deforestRasterArray[::100].flatten()# select subset of samples and flatten it 1d
        return deforestRasterArray


def processPopulation(): # function for preprocessing population raster
        
    populationRasterPath='F:/myenv/popproj.tif'
    with rasterio.open(populationRasterPath) as population:
         populationArray=population.read(1)
         profile=population.profile
    binaryPopulationArrayPath='D:/myenv/popProjBinary.tif' 

    binaryPopulationArray=np.where(populationArray>0,1,0)# create binary population array
     
    with rasterio.open(binaryPopulationArrayPath, 'w', **profile) as dst:
            dst.write(binaryPopulationArray[::10], 1)     
    
    populationRasterDistancePath= "D:/myenv/popdist.tif"
    

    computeDistance(binaryPopulationArrayPath,populationRasterDistancePath) # computes the population distance raster

    ##################### Creating topo raster array with same dimensions as deforest raster#################################### 
        
    newPopulationArray = np.zeros((46396,47723), dtype=populationArray.dtype)

    # Copy the original raster array into the new array
    newPopulationArray[:populationArray.shape[0], :populationArray.shape[1]] =populationArray 
    newPopulationArray=newPopulationArray[::10].flatten()

    # Fill the remaining areas with a specific value (e.g., 0)
    
    newPopulationArray[populationArray.shape[0]:, :] = 0
    newPopulationArray[:, populationArray.shape[1]:] = 0
    return newPopulationArray
################################################################################################################################

      
def processRoadDistance(): # function for processing road distance raster
    
    roadRaster= "F:/myenv/roadsdist/roadraster.tif" # road raster  output folder
    createRoadRaster()
    roadrasterDistancePath ='D:/myenv/roaddistraster.tif'
    computeDistance(roadRaster,roadrasterDistancePath)
   
    roads=gdal.Open(roadrasterDistancePath)
    roadDistArray=roads.GetRasterBand(1).ReadAsArray()[::100].flatten()
    del roads     
    return roadDistArray

def processTopo(): # function for processing topography raster
    topo=gdal.Open('F:/myenv/topoData.tif')
    topoArray=topo.GetRasterBand(1).ReadAsArray()
             

##################### Creating topo raster array with same dimensions as deforest raster#################################### 
    # Create a new array with the desired dimensions filled with a specific value
    newTopoArray = np.zeros((46396,47723), dtype=topoArray.dtype)

    # Copy the original raster array into the new array
    newTopoArray[:topoArray.shape[0], :topoArray.shape[1]] =topoArray 

    # Fill the remaining areas with a specific value (e.g., 0)
    
    newTopoArray[topoArray.shape[0]:, :] = 0
    newTopoArray[:, topoArray.shape[1]:] = 0

    newTopoArray=newTopoArray[::100].flatten()# select subset of samples and flatten it to 1d
    #####################################################################################################################
    print('topogreaterzero',np.sum(newTopoArray>1))
    
    print('newTopoArrayshape',newTopoArray.shape)
    return newTopoArray

def processCrops(): # function for processing crops landuse raster
    with rasterio.open('F:/myenv/luproj.tif') as landUse:
         cropsArray=landUse.read(1) #read the crops landuse band
         profile=landUse.profile
    binaryCropsArray=np.where(cropsArray>0,1,0)# create binary crops landuse array
    del cropsArray
            
    binaryCropsArrayPath='D:/myenv/cropsProjBinary.tif'  
    with rasterio.open(binaryCropsArrayPath, 'w', **profile) as dst:
            dst.write(binaryCropsArray, 1)     
    
    cropsRasterDistancePath= "D:/myenv/cropsdist.tif"
    

    computeDistance(binaryCropsArrayPath,cropsRasterDistancePath) # computes the  distance raster from crops

       
    cropsDist=gdal.Open(cropsRasterDistancePath)
    cropsDistArray=cropsDist.GetRasterBand(1).ReadAsArray()
   
     
    #################### Creating crop  landuse raster arrays with same dimensions as deforest raster#################################### 
    # Create a new array with the desired dimensions filled with a specific value
    newCropsArray = np.zeros((46396,47723), dtype=cropsDistArray.dtype)

    # Copy the original raster array into the new array
    newCropsArray[:cropsDistArray.shape[0], :cropsDistArray.shape[1]] =cropsDistArray

    # Fill the remaining areas with a specific value (e.g., 0)
    
    newCropsArray[newCropsArray.shape[0]:, :] = 0
    newCropsArray[:,newCropsArray.shape[1]:] = 0
    
    
    return newCropsArray

def processUrban(): # function for processing landuse raster
    with rasterio.open('F:/myenv/luproj.tif') as landUse:
         urbanArray=landUse.read(2)# read the urban landuse band
         profile=landUse.profile
    binaryUrbanArray=np.where(urbanArray>0,1,0)
    del urbanArray
    binaryUrbanArrayPath='D:/myenv/urbanProjBinary.tif'  # create binary urban landuse array
    with rasterio.open(binaryUrbanArrayPath, 'w', **profile) as dst:
            dst.write(binaryUrbanArray, 1)     
    
    urbanRasterDistancePath= "D:/myenv/urbandist.tif"
     
    computeDistance(binaryUrbanArrayPath,urbanRasterDistancePath) # computes the urban landuse distance raster
       

    urbanDist=gdal.Open(urbanRasterDistancePath)
    urbanDistArray=urbanDist.GetRasterBand(1).ReadAsArray()
    
    ##################### Creating urban landuse raster arrays with same dimensions as deforest raster#################################### 
        
    newUrbanArray = np.zeros((46396,47723), dtype=urbanDistArray.dtype)

    # Copy the original raster array into the new array
    newUrbanArray[:urbanDistArray.shape[0], :urbanDistArray.shape[1]] =urbanDistArray

    # Fill the remaining areas with a specific value (e.g., 0)
    
    newUrbanArray[newUrbanArray.shape[0]:, :] =0
    newUrbanArray[:,newUrbanArray.shape[1]:] = 0

    newUrbanArray=newUrbanArray[::100].flatten()# select subset of samples and flatten it to column vector
    #####################################################################################################################
    
    return newUrbanArray    
     


def rfTrainingAndPredictions(): # function for training and making predictions using random forest model

    global deforestRasterArray
    # Convert the column array to a DataFrame column
    df = pd.DataFrame({'Population':processPopulation(),'Urban':processUrban(),'RoadDistance':processRoadDistance(),'Topography':processTopo(),'Labels':processDeforestMap()})
    
   
    #Split the data into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(df[['RoadDistance','Topography','Crops','Urban','Population']], df['Labels'], test_size=0.2, random_state=42)
           
    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(xTrain,yTrain)
     
    # Predict probabilities for each class
    predictedProbabilities = model.predict_proba(xTest)
    predictedProbabilities=predictedProbabilities[:,1]# selecting  class 1 only i.e deforestation
        
    deforestRiskMapPath='D:/myenv/deforestriskmap.tif'
    with rasterio.open(deforestRiskMapPath, 'w', **profile) as dst:
            dst.write(predictedProbabilities, 1) # creating the map

    
rfTrainingAndPredictions()
