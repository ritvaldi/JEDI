# Base class for sentinel reader
"""
Classes for reading in sentinel files from the .SAFE format and outputting as
np arrays.
"""
from convertbng.util import convert_lonlat # BNG to Lat Lon conversion
import datetime as dt
import geojson # handles geojson import
import glymur # handles jp2 import
import mgrs # MGRS and UTM to Lat Lon conversion
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PIL import Image
# from past.utils import old_div # part of futures can remove in future as is 
# not needed in current version
import shapefile # handles shp file import
from scipy.ndimage import zoom # interpolation
from skimage import draw
import xml.etree.ElementTree as ET # XML handling
import warnings

# suppress warnings for now
warnings.filterwarnings('ignore')

# Abstract base class to test the file type.
class sentinelTwoImage(object):
    """
    The base unit for the sentinelFile class.
    User Methods:
    pixel_as_LonLat -- returns a pixel value in Lon Lat coordinates
    
    Arguments:
    filePath -- the path to the .jp2 image file
    date -- a datetime object of the captureDate
    mgrsGridSquare -- string in the format 'XXYZZ' where XX is UTM longitude,
                      Y is UTM latitude and ZZ is the grid square. E.g. 30UXE
    processLevel -- string (either 'Level-1c' or 'Level-2Ap')
    getGEOref -- logical. If true will attempt to access pixel georef from the 
                 .jp2 header box.
    
    Attributes:
    refPixUtm -- UTM coordinates of top left pixel
    
    """
    def __init__(self,filePath,date,mgrsGridSquare,processLevel,bandName,getGEOref=True):
        # pull some info from the name string
        self.filePath = os.path.abspath(filePath)
        self.processLevel = processLevel
        self.captureDate = date
        self.mgrsGridSquare = mgrsGridSquare
        self.getGEOref = getGEOref
        self.bandName = bandName
        self.data = self._read_image()
        # get additional file info
        self._get_info()
    
    def _read_image(self):
        """ returns the image associated with the instance"""
        im = glymur.Jp2k(self.filePath)
        return im
    
    def _get_info(self):
        """
        Internal method for extracting the pixel georeferencing and resolution
        """
        # read image to get size and geo info.
        if self.getGEOref == True:
            self._get_image_georef(self.data)
        # get resolution from pix width
        pixWidth = self.data.shape[0]
        self.pixelSize = int(109800/pixWidth)
            
    # have replaced this with a new version which also stores the bounding box
    def _get_image_georef(self,im):
        """ Gets georeference of top left pixel of sentinel .jp2
        """
        # get the image georeferencing from the jp2 image header
        xmlString = im.box[3].box[1].box[1].__str__()
        # find the gml tag corresponding to position
        posStart = xmlString.find('<gml:pos>')+9
        posStop = xmlString.find('</gml:pos>')
        refPix = xmlString[posStart:posStop].split()
        self.refPixUtm = (int(refPix[0]),int(refPix[1]))
        # get the lon band and hemisphere from the mgrs grid square
        m = mgrs.MGRS()
        utm = m.MGRSToUTM(self.mgrsGridSquare.encode('utf-8'))
        self.refPixUtmZone = utm[0]
        self.refPixUtmHemisphere = utm[1].decode('utf-8')
        # generate UTM coordinates for 4 corners of bounding box
        # make tuple of coordinates describing the bounding polygon.
        XYs_utm = (self.refPixUtm,
                  (self.refPixUtm[0]+100000,self.refPixUtm[1]),
                  (self.refPixUtm[0]+100000,self.refPixUtm[1]-100000),
                  (self.refPixUtm[0],self.refPixUtm[1]-100000),
                  self.refPixUtm)
        self.bboxUtm = XYs_utm
        

# this subclass handles the standard MSI format data (single layer)    
class msiFile(sentinelTwoImage):
    """Class for Multi Spectral Image sentinel files.
    
       Keyword Arguments:
       sentinelFile -- path to sentinel .jp2 band image file
       
       User Methods:
    """
    def __repr__(self):
        return('\nMulti-Spectral Image Slice\n---------------------------\nFile location: ' \
               +self.filePath+ \
               '\n\nPixel Resolution: '+str(self.pixelSize)+'m\n' \
               'Band Name: '+self.bandName+'\n'+
               'Date: '+str(self.captureDate)+'\n')
        
    

# this subclass handles the SCL format data            
class sclFile(sentinelTwoImage):
    """Class for Scene Classification Layer sentinel files.
    
       Keyword Arguments:
       sentinelFile -- path to sentinel .jp2 SCL file
       
       User Methods:
       
       Attributes:
       layer_names -- list of band names corresponding to layer
       
    """
    layerNames = {0:'NODATA',
                   1:'SATURATED_DEFECTIVE',
                   2:'DARK_FEATURE_SHADOW',
                   3:'CLOUD_SHADOW',
                   4:'VEGETATION',
                   5:'BARE_SOIL_DESERT',
                   6:'WATER',
                   7:'CLOUD_LOW_PROBA',
                   8:'CLOUD_MEDIUM_PROBA',
                   9:'CLOUD_HIGH_PROBA',
                   10:'THIN_CIRRUS',
                   11:'SNOW_ICE'}
    
    def __repr__(self):
        return('\nScene Classification Layer\n---------------------------\nFile location: ' \
               +self.filePath+ \
               '\n\nPixel Resolution: '+str(self.pixelSize)+'m\n' \
               'Band Name: '+self.bandName+'\n'+
               'Date: '+str(self.captureDate)+'\n')

#  
class sentinelTwoGranule(object):
    """
    Base class for reading a sentinel 2 SAFE folder
    
    Keyword arguments:
    SAFE_folder_path -- path to the .SAFE directory
    mgrsGridSquare -- the string corresponding to the MGRS coordinates of the
                    desired granule.
    ROI -- a valid sentinelROI object
    either a mgrsGridSquare for the desired granule (e.g. 'T30UXC') or a 
    sentinelROI object to specify the granule of interest.
    
    There are no user methods for this class.
    """
    # ROI should be supplied as a sentinelROI object 
    def __init__(self,safeFolderPath,mgrsGridSquare=None,ROI=None):
        if mgrsGridSquare == None and ROI == None:
            raise ValueError('Requires either a MGRS search string (e.g. 30UXC)\
                   or a ROI object')
        self.path = safeFolderPath
        # useful for batch processing and avoiding errors.
        self.include = False
        # get top level xml data
        self.topXML = self._openXML(self._findTopXml())
        # check processing type
        self._getProcessLevel()
        
        # convert to 1c or 2a class
        if self.processLevel == 'Level-1C':
            sentinelTwoGranule_1C.convert_to_1C(self)
        elif self.processLevel == 'Level-2Ap':
            sentinelTwoGranule_2A.convert_to_2A(self)
        else:
            raise TypeError('Unsupported product type')
            
        # if ROI supplied, use instead of granule search string
        if ROI:
            self.mgrsGridSquare = self._make_granule_searchstring(ROI)
        else:
            self.mgrsGridSquare = mgrsGridSquare
        # find the correct granule
        self._findGranule(self.mgrsGridSquare)
        self._getDate()
        self._get_bands()
    
    # preview method has been replaced with imshow in order to bring in line
    # with standard pythonic conventions

    def imshow(self):
        """
        Plots preview image with MPL
        """
        dir1 = os.path.join(self.granulePath,'QI_DATA')
        for f in os.listdir(dir1):
            if f.endswith('.jp2') and f.count('PVI')>0:
                plt.figure(figsize=(6,6))
                im = glymur.Jp2k(os.path.join(dir1,f))[:]
                return plt.imshow(im)
    
                                         
    def __repr__(self):
        return('\nGeneric SAFE Package\n---------------------------\nFile location: ' \
               +self.path+ \
               '\n Date: '+str(self.captureDate))

    # returns a path for the granule directory within the SAFE file
    def _make_granule_searchstring(self,ROI):
        searchstring = str(ROI.mgrs_lon[0])+str(ROI.mgrs_lat[0])+str(ROI.mgrs_square[0])
        return searchstring
    
    # function to find the top level XML from a ZIP file
    def _findTopXml(self):
        for file in os.listdir(self.path):
            # look for xml file extension and 'MTD' string in name
            if file.endswith('.xml') and file.find('MTD_') > -1:
                topXML = os.path.join(self.path,file)
        return topXML
    
    # function to extract top level xml file to a tree/root structure
    def _openXML(self,xmlPath):
        metaTree = ET.parse(xmlPath)
        return metaTree.getroot()
    
    # function to return granule list from top XML
    def _returnGranuleList(self):
        roo = self.topXML
        granules = []
        # search xml for granule list and extract.
        for granule in roo.findall(".//Granule_List/"):
            granules.append(granule.get("granuleIdentifier"))
        if len(granules) < 1:
            raise ValueError("No Granules found")
        return granules
    # find granules
    def _findGranule(self,searchstring):
        # get the SAFE type
        self._getProductFormat()
        # get a list of all granuleIdentifiers
        granules = self._returnGranuleList()
        matches = []
        for granule in granules:
            if granule.find(searchstring) > -1:
                matches.append(granule)
        
        # control the include switch variable
        # can be 3 copies of the same granule because 3 different resolutions in
        # level 2a
        if len(matches) > 3:
            self.include = False
            raise ValueError('Too many granules with the same name in directory')
        elif len(matches) == 0:
            self.include = False
            raise ValueError('No granules found')
        # Take first item as granule Identifier
        granuleIdentifier = matches[0]
        
        # check safe type and use appropriate method for making an image path
        root = self.topXML
        if self.productFormat == 'SAFE_COMPACT':
            # use the image_file path within the safe xml
            longpath = root.findall(".//Granule_List/Granule/[@granuleIdentifier=\'"+\
                                    granuleIdentifier+"\']/")[0].text
            #split into components
            lp = longpath.split('/')
            # make an image path
            self.imagePath = os.path.join(self.path,lp[0],lp[1],lp[2])
            self.granulePath = os.path.join(self.path,lp[0],lp[1])
        # for safe format, just use the path directly
        elif self.productFormat == 'SAFE':
            self.imagePath = os.path.join(self.path,'GRANULE',matches[0],'IMG_DATA')
            self.granulePath = os.path.join(self.path,'GRANULE',matches[0])
            
    # return the product format
    def _getProductFormat(self):
        root = self.topXML
        self.productFormat = root.findall(".//PRODUCT_FORMAT")[0].text
    
    # method for retrieving process level from xml.
    def _getProcessLevel(self):
        self.processLevel = self.topXML.findall(".//PROCESSING_LEVEL")[0].text
                                               
    # method for retrieving capture Time.
    def _getDate(self):
        datestr = self.topXML.findall(".//PRODUCT_START_TIME")[0].text
        self.captureDate = dt.date(year = int(datestr[0:4]),\
                                   month = int(datestr[5:7]),\
                                   day = int(datestr[8:10]))
        
class sentinelTwoGranule_1C(sentinelTwoGranule):
    
    """ subclass for sentinel 1C granule

    """
    def __repr__(self):
        return('\nSentinel Level 1C SAFE Package\n---------------------------\nFile location: ' \
               +self.path+ \
               '\nDate: '+str(self.captureDate))
    
    
            
    def _get_bands(self):
        # Image folder paths
        bands = ['B01','B02','B03','B04','B05','B06','B07','B8A','B09','B10'\
                 'B11','B12']
        self.msiBands = {}
        # import 10m bands
        for file in os.listdir(self.imagePath):
            for band in bands:
                if file.find(band) > -1:
                    self.msiBands[band] = msiFile(os.path.join(self.imagePath,file),\
                                                  self.captureDate,
                                                  self.mgrsGridSquare,
                                                  self.processLevel,
                                                  band,
                                                  getGEOref=True)
    @classmethod
    def convert_to_1C(cls, obj):
        obj.__class__ = sentinelTwoGranule_1C
        
class sentinelTwoGranule_2A(sentinelTwoGranule):
    
    """subclass for sentinel 2A granule
    """
    def __repr__(self):
        return('\nSentinel Level 2A SAFE Package\n---------------------------\nFile location: ' \
               +self.path+ \
               '\nDate: '+str(self.captureDate))
    # get all band files. Will try and get only the highest res images.
    # have defined it all explicitly for now, for readability.
    def _get_bands(self):
        
        # Image folder paths
        self.R10mPath = os.path.join(self.imagePath,'R10m')
        self.R20mPath = os.path.join(self.imagePath,'R20m')
        self.R60mPath = os.path.join(self.imagePath,'R60m')
        self.msiBands = {}
        # 10m bands first
        bands10m = ['B02','B03','B04','B08']
        bands20m = ['B05','B06','B07','B8A','B11','B12']
        bands60m = ['B01','B09']
        
        # import 10m bands
        for file in os.listdir(self.R10mPath):
            for band in bands10m:
                if file.find(band) > -1:
                    self.msiBands[band] = msiFile(os.path.join(self.R10mPath,file),
                                                  self.captureDate,
                                                  self.mgrsGridSquare,
                                                  self.processLevel,
                                                  band,
                                                  getGEOref=True)
        # 20m bands
        for file in os.listdir(self.R20mPath):
            for band in bands20m:
                if file.find(band) > -1:
                    self.msiBands[band] = msiFile(os.path.join(self.R20mPath,file),
                                                  self.captureDate,
                                                  self.mgrsGridSquare,
                                                  self.processLevel,
                                                  band,
                                                  getGEOref=True)
        # 60m bands
        for file in os.listdir(self.R60mPath):
            for band in bands60m:
                if file.find(band) > -1:
                    self.msiBands[band] = msiFile(os.path.join(self.R60mPath,file),
                                                  self.captureDate,
                                                  self.mgrsGridSquare,
                                                  self.processLevel,
                                                  band,
                                                  getGEOref=True)
        # import Scene classifier layer file
        if self.productFormat == 'SAFE':
            im_dir = self.imagePath
        elif self.productFormat == 'SAFE_COMPACT':
            im_dir = self.R20mPath
        for file in os.listdir(im_dir):
            # check it's the 20m version
            if file.find('_SCL_') > -1 and file.find('20m') > -1:
                self.SCL = sclFile(os.path.join(im_dir,file),
                                   self.captureDate,
                                   self.mgrsGridSquare,
                                   self.processLevel,
                                   'Scene Classification Layer',
                                   getGEOref=True)
                
    @classmethod
    def convert_to_2A(cls, obj):
        obj.__class__ = sentinelTwoGranule_2A
        
    
        
class sentinelROI(object):
    
    """
    Class for converting coordinate strings between coordinate systems.
    Input files must contain only a single ROI at present
    Keyword Arguments:
    path -- path to a coordinate file.
    file_type -- text string 'kml' or 'geojson' at present. Default is
    None and so looks for file extension instead.
    feature -- default 0 - the feature within the geojson file. Ignored for
    kml at present.
    """

    def __init__(self, path=None, file_type=None, feature_n=0):
        self.lat = []
        self.lon = []
        self.mgrsValue = []
        self.mgrsLat = []
        self.mgrsLon = []
        self.mgrsSquare = []
        self.coordinates_utm = []
        # adding a dictionary to include any useful metadata from files
        self.records = {}
        # explicit setting of file type overides file extension
        if path:
            if file_type:
                self.fileType = file_type
            else:
                self.fileType = path[-3:]
                    
            # get the map file
            if self.fileType == 'kml':
                self._coords_from_kml(path)
            elif self.fileType == 'son' or \
                self.fileType == 'geojson' or \
                self.fileType == 'GEOjson': #geojson
                self._coords_from_geojson(path,feature_n)
            elif self.fileType == 'shp' or \
                self.fileType == 'shx' or \
                self.fileType == 'dbf':
                    self._coords_from_shp(path,'BNG',feature_n)
            else:
                raise IOError('unrecognised geo file')
            # make coordinates attribute
            self._leaflet_coords()
            # extract the ROI lat/lon
            self._convert_to_grid()
            # calculate bouding box coordinates
            self._boxROI_utm()
    
    def _coords_from_kml(self,kml_file):
        # parse kml as an xml
        metaTree = ET.parse(kml_file)
        root = metaTree.getroot()
        prefix = root[0][0].tag[:-4] # removes the 'name' string to get the tag prefix
        # check is a kml
        if prefix.find('/kml/') > -1:
            string = './/'+prefix+'coordinates'
            coord_list = root.findall(string)[0].text.split()
            for tripletStr in coord_list:
                triplet = tripletStr.split(',')
                self.lon.append(float(triplet[0]))
                self.lat.append(float(triplet[1]))
        
    # from the sentinel sat package
    # remember to convert back to %.7f if values to be passed to copernicus
    def _coords_from_geojson(self,geojson_file,feature_number):
        geojson_obj = geojson.loads(open(geojson_file, 'r').read())
        coordinates = geojson_obj['features'][feature_number]['geometry']['coordinates'][0]
        for pair in coordinates:
            self.lon.append(float(pair[0]))
            self.lat.append(float(pair[1]))
            
    # using the shapefile package. Only making BNG importer at the moment
    def _coords_from_shp(self,shp_file,coord_system,feature_number,getRecords=True):
        """ Imports British Grid shape files as LonLat coords.
        """
        if coord_system == 'BNG':
            sf = shapefile.Reader(shp_file)
            BNGcoordinates = sf.shapes()[feature_number].points
            # split coords, convert and append to self.lat and self.lon
            for pair in BNGcoordinates:
                lola = convert_lonlat(pair[0],pair[1])
                self.lon.append(float(lola[0][0]))
                self.lat.append(float(lola[1][0]))
        # get record entries and append to dictionary
        if getRecords==True:
            for i in range(1,len(sf.fields)):
                self.records[sf.fields[i][0]] = sf.records()[feature_number][i-1]
            
    # function for converting lat/lon to MGRS + UTM grid systems
    def _convert_to_grid(self):
        for la,lo in zip(self.lat,self.lon):
            m = mgrs.MGRS()
            mgrsCoord = m.toMGRS(la,lo).decode('utf-8')
            self.mgrsValue.append(mgrsCoord)
            self.mgrsLon.append(mgrsCoord[:2])
            self.mgrsLat.append(mgrsCoord[2:3])
            self.mgrsSquare.append(mgrsCoord[3:5])
            UTMcoord = m.MGRSToUTM(mgrsCoord.encode('utf-8'))
            self.coordinates_utm.append((UTMcoord[2],UTMcoord[3]))
            
        if self.is_one_square() == True:
            self.mgrsGridSquare = str(self.mgrsLon[0])+self.mgrsLat[0]+self.mgrsSquare[0]
        else:
            self.mgrsGridSquare = -1
        # won't work if ROI is in more than one
    
    # function to check ROI lies in only one grid square        
    def is_one_square(self):
        """Returns True if ROI covers only a single grid square"""
        sq = self.mgrsSquare
        if sq.count(sq[0]) == len(sq):
            return True
        else:
            return False
    # return UTM coordinates for bounding box
    def _boxROI_utm(self):
        minx = min([x[0] for x in self.coordinates_utm])
        maxx = max([x[0] for x in self.coordinates_utm])
        miny = min([x[1] for x in self.coordinates_utm])
        maxy = max([x[1] for x in self.coordinates_utm])
        
        # makes a coordinate box in the same format as the image file
        self.bboxUtm = ((minx,maxy),
                        (maxx,maxy),
                        (maxx,miny),
                        (minx,miny),
                        (minx,maxy))
    
    def _leaflet_coords(self):
        self.coordinates = []
        for la,lo in zip(self.lat,self.lon):
            coords = (la,lo)
            self.coordinates.append(coords)
        
class batchProcessor(object):
    """
    processor for processing lists of rois on a single granule 
    """
    
    # define some useful parameters
    # set up blank variable to hold the arrays if they're contained within
        # the class
    arrays = None
    masks = None
    pix10 = None
    pix20 = None
    all = False
    # this really speeds up processing as minimises file opening operations
    # no need to specify granules or rois on initialisation
    def __init__(self,granule=None,listOfRois=None):
        # if ROIs supplied, do some geo checking
        if type(listOfRois==list):
            first = listOfRois[0]
            # check all rois in list match the first one
            T = [is_geomatch(first,roi) for roi in listOfRois]
            # if any do not, raise an error
            if T.count(False) > 0:
                raise ValueError('ROIs must all be in same grid square')
            else:
                self.roiList = listOfRois
                
        # if granule supplied, do some more geo checking
        if granule != None and type(listOfRois==list):
            if is_geomatch(granule,listOfRois[0]) != True:
                raise ValueError('ROI and granule are not in same grid square')
            elif type(granule) != sentinelTwoGranule_2A:
                raise TypeError('batchProcessor class only implemented for \
                sentinelTwoGranule_2A')
            else:
                self.granule = granule
        
        # set up user variables for the bands
        # these are lists not dictionaries as easier to retrieve
        self.bandNames = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
        self.index = [0,1,2,3,4,5,6,7,8,9]
        self.bandCentres = [496.6,560.0,664.5,703.9,740.2,782.5,835.1,864.8,1613.7,2202.4]
            
    def make_arrays(self, storeArrays=True, waveBand='all', mask=False,
                    excludedLayers=None):
        """ returns a list of m x n x lambda arrays (len(list) == len(listOfROis))
        """
        # do some cleanup
        self.arrays = None
        self.masks = None
        self.pix10 = None
        self.pix20 = None
        # lets ignore the 60m bands for now - probably not useful and makes the
        # generalised case harder
        
        # only does the pixel calculations if not already done
        self._make_extract_lists()
        # setup the band info
        bands10m = ['B02','B03','B04','B08']
        bands20m = ['B05','B06','B07','B8A','B11','B12']
        # find the limits at each resolution
        l10 = self._find_image_open_limits(self.pix10)
        l20 = self._find_image_open_limits(self.pix20)
        # store l20 for later
        self.l20 = l20
        
        # if all bands wanted
        if waveBand == 'all':
            self.all = True
            
            # indexes for positioning the bands in the np array                    
            i10 = [0,1,2,6] 
            i20 = [3,4,5,7,8,9] 
            output = []
            
            # setup blank nparrays
            for roi in self.pix10:
                # create an empty array and put in output list
                y = roi[1]-roi[0]
                x = roi[3]-roi[2]
                output.append(np.empty((y,x,10)))
        
            # get the 10m bands
            for band,ind in zip(bands10m,i10):
                # get whole image as nparray
                image = np.asarray(
                        self.granule.msiBands[band].data[l10[0]:l10[1],l10[2]:l10[3]])
                
                for i, roi in enumerate(self.pix10):
                    # put the extracted region into the correct position in the output np array
                    
                    output[i][:,:,ind] = image[roi[0]-l10[0]:roi[1]-l10[0],roi[2]-l10[2]:roi[3]-l10[2]]
            
            # 20m bands
            for band,ind in zip(bands20m,i20):
                # get whole image as nparray
                image = np.asarray(
                        self.granule.msiBands[band].data[l20[0]:l20[1],l20[2]:l20[3]])
                
                for i, roi in enumerate(self.pix20):
                    # put the extracted region into the correct position in the output np array
                    im = image[roi[0]-l20[0]:roi[1]-l20[0],roi[2]-l20[2]:roi[3]-l20[2]]
                    # do the interpolation upscaling
                    im = zoom(im,2,order = 1) # order 1 is bilinear - should be more robust to NaNs
                    output[i][:,:,ind] = im 
        
        # single band example
        elif type(waveBand) == str:
            self.all = False
            output = []
            # get the appropriate band
            outBand = self.granule.msiBands[waveBand]
            if outBand.pixelSize == 10:
                # get whole image as nparray
                image = np.asarray(
                        outBand.data[l10[0]:l10[1],l10[2]:l10[3]])
                for roi in self.pix10:
                    # put the extracted region into the correct position in the output np array
                    output.append(
                            image[roi[0]-l10[0]:roi[1]-l10[0],roi[2]-l10[2]:roi[3]-l10[2]]
                            )
                    
            elif outBand.pixelSize == 20:
                # get whole image as nparray
                image = np.asarray(
                        outBand.data[l20[0]:l20[1],l20[2]:l20[3]])
                for i,roi in enumerate(self.pix20):
                    # put the extracted region into the correct position in the output np array
                    output.append(
                            zoom(
                            image[roi[0]-l20[0]:roi[1]-l20[0],roi[2]-l20[2]:roi[3]-l20[2]],
                            2,
                            order = 3)
                            )
                            
        # deal with any masking
        if mask == True:
            masks = []
            # iterate through the roiList and make a mask for each one
            for roi,ref,bbox in zip(self.roiList,self.newRefPixUtm,self.pix10):
                masks.append(self._make_mask(roi,ref,bbox)-1)
            # if scene classifier masking specified
            if excludedLayers != None:
                # retrieve list of masks
                sclMasks = self._make_scl_masks(l20,excludedLayers)
                # combine the masks
                masks = [np.array(np.logical_or(m1,m2)-1,dtype='bool') for m1,m2 in zip(masks,sclMasks)]
                self.masks = masks

            # iterate through ROIs, convert to masked array and add mask
            for i in range(len(output)):
                # reshape to X x Y x 10
                m = masks[i].reshape(masks[i].shape[0],masks[i].shape[1],1).repeat(10,axis=2)
                array_ = output[i]
                output[i] = np.ma.array(array_,mask=m-1)
                
        if storeArrays == True:
            self.arrays = output
            return None
        else: 
            return output

    def as_NP_arrays(self):
        """ returns arrays variable """
        if self.arrays is None:
            raise NameError('No arrays to return')
        return self.arrays
        
    def combine_ROIs(self):
        """ combines all ROIs in instance to a single ROI
        """
        roiList = self.roiList
        
        lats = []
        lons = []
        for r in roiList:
            for c in r.coordinates:
                # if cLat < minLat, replace minLat
                lats.append(c[0])
                lons.append(c[1])
        minLat = min(lats)
        maxLat = max(lats)
        minLon = min(lons)
        maxLon = max(lons)
        newROI = sentinelROI()
        newROI.lat = [maxLat,maxLat,minLat,minLat,maxLat]
        newROI.lon = [minLon,maxLon,maxLon,minLon,minLon]
        # make coordinates attribute
        newROI._leaflet_coords()
        # extract the ROI lat/lon
        newROI._convert_to_grid()
        # calculate bouding box coordinates
        newROI._boxROI_utm()
        self.roiList=[newROI]
        # run the make arrays again
        self.make_arrays()

    # this is the only convenience function in the class 
    def as_RGB_images(self, mask = False):
        """ returns a list of RGB PIL.Image (len(list) == len(listOfROis))
        """
        # read each channel in if not already read
        if self.arrays != None and self.all == True:
            bs = [x[:,:,0] for x in self.arrays]
            gs = [x[:,:,1] for x in self.arrays]
            rs = [x[:,:,2] for x in self.arrays]
        else:
            raise ValueError('No arrays to output. Run make_arrays() first')
        
        output = []
        # import the image profile from file
        cdf,bins = pickle.load(open('imProfile.p', 'rb'))
        length = np.arange(0,len(self.pix10))
        # iterate through ROIs and make images
        for i,roi,ref,bbox,r,g,b in zip(length,
                                        self.roiList,
                                        self.newRefPixUtm,
                                        self.pix10,
                                        rs,
                                        gs,
                                        bs):
            
            stack = np.stack([r,g,b]).transpose(1,2,0)
            # apply a histogram equalisation
            rgb,_,_ = image_histogram_equalization(stack, cdf=cdf, bins=bins)
            rgb = rgb.astype('uint8')
            
            # handle the masking
            if mask == True:
                m = self.masks[i]
                rgb = rgb*np.dstack((m,m,m))
            output.append(Image.fromarray(rgb))
        return output
    
    # this is currently a really crude version which just takes an average over
    # each np array
    def as_mean_spectrum(self):
        """ returns an n x lambda np array for n rois
        """
        
        # see if the pixel extraction has already been done
        
        if self.all == False:
            self.make_arrays(storeArrays=True)
            
        output = []
        
        for array in self.arrays:
            output.append(np.mean(array,axis = (0,1)))
        
        return(output)
    
    def _make_scl_masks(self,l20,layers):
        """generates a list of masks from the scl_layers"""
        # l20 is the 20m mask limits
        # read in the scl image
        image = np.asarray(self.granule.SCL.data[l20[0]:l20[1],l20[2]:l20[3]])
        output = []
        for i, roi in enumerate(self.pix20):
            # put the extracted region into the correct position in the output np array
            im = image[roi[0]-l20[0]:roi[1]-l20[0],roi[2]-l20[2]:roi[3]-l20[2]]
            # do the interpolation upscaling
             # order 1 is bilinear - should be more robust to NaNs
             #im = np.ceil(zoom(im,2,order = 1))
            # using a nearest interpolation as boolean
            im = zoom(im,2,mode='nearest')
            # set up a mask with the first layer
            mask = im == layers[0]
            # if multiple layers
            if len(layers) > 1:
                for i in range(1,len(layers)):
                    # perform the logical or operation between different masks
                    # and do the zoom
                    mask = np.logical_or(mask,im==layers[i])
            output.append(mask)
        return output
    
    def _find_image_open_limits(self,pixList):
        # takes the full list of pixCoords and returns the bounding coordinates
        minY = min([y[0] for y in pixList])
        maxY = max([y[1] for y in pixList])
        minX = min([x[2] for x in pixList])
        maxX = max([x[3] for x in pixList])
        return (minY,maxY,minX,maxX)
       
    def _make_extract_lists(self):
        """ does all the UTM to pixel conversion
        """
        # calculate the corners of the 20m safe region
        # Then extract a region in the 10m based on the pixel coords above
        refPix10 = self.granule.msiBands['B02'].refPixUtm
        refPix20 = self.granule.msiBands['B05'].refPixUtm
    
        pix20 = [] # pixel coordinates of bbox at 20m res (polygon form)
        pix10 = [] # pixel coordinates of bbox at 10m res (polygon form)
        self.newRefPixUtm = [] # utm coordinates of 10m pixel
        # iterate through the ROIs
        
        for roi in self.roiList:
            # calculate the 20m pixels first
            r20 = self._bbox_to_pixels(refPix20,roi.bboxUtm,20)
            r10 = [tuple([n*2 for n in m]) for m in r20]
            pix20.append(self._min_max_vals(r20))
            pix10.append(self._min_max_vals(r10))
            # write the new refPix for each ROI
            self.newRefPixUtm.append(self._pixel_to_utm(refPix10,
                          (min([x[0] for x in r10]),min([y[1] for y in r10])),
                          10))
        
        self.pix20 = pix20
        self.pix10 = pix10
                
                          
    def _min_max_vals(self,listOfCoords):
        """ takes any list of pix coords and returns array format min/max (bbox)
        """
        X = [x[0] for x in listOfCoords]
        Y = [x[1] for x in listOfCoords]
        
        # returns in array form
        return (min(Y),max(Y),min(X),max(X))
        
    def _pixel_to_utm(self,refPixUtm,pixel,pixSize):
        refx = refPixUtm[0]
        refy = refPixUtm[1]
        px = pixel[0]
        py = pixel[1]
        newx = refx+(px*pixSize)
        newy = refy-(py*pixSize)
        
        return(int(round(newx)),int(round(newy)))
    
    def _bbox_to_pixels(self,refPixUtm,coordList,pixSize):
        
        exact = [self._utm_to_pp(refPixUtm,coord,pixSize) for coord in coordList]
            
        minx = min([x[0] for x in exact])
        maxx = max([x[0] for x in exact])
        miny = min([x[1] for x in exact])
        maxy = max([x[1] for x in exact])
            
        rounded = []
        
        for point in exact:
            
            # round x vals
            if point[0] == minx:
                # round down to expand ROI towards ref point
                xr = self._rounder(point[0],roundUp=False,base=1)
            elif point[0] == maxx:
                # round up to expand ROI away from ref point
               xr = self._rounder(point[0],roundUp=True,base=1)
               
            if point[1] == miny:
                # round down to expand ROI towards ref point
                yr = self._rounder(point[1],roundUp=False,base=1)
            elif point[1] == maxy:
                # round up to expand ROI away from ref point
               yr = self._rounder(point[1],roundUp=True,base=1)
               
            rounded.append((xr,yr))
        
        return rounded

    
    def _utm_to_pp(self,refPixUtm,pixUtm,pixSize):
        x_exact = (pixUtm[0]-refPixUtm[0])/float(pixSize)
        y_exact = -(pixUtm[1]-refPixUtm[1])/float(pixSize)
        return (x_exact,y_exact)
    
    # function for controlling rounding to nearest factor
    def _rounder(self,val,roundUp,base):
        base = int(base)
        if roundUp == False:
            newVal = (val//base)*base
        elif roundUp == True:
            newVal = ((val//base)*base)+base
        elif roundUp == 'Nearest':
            newVal = round(val/base)*base
        return int(newVal)

    def _make_mask(self,ROI,refPixUtm,bbox):
        """ 
        Returns a boolean mask array from a ROI, based on the original image

        """
        # get current pixel scale will always be 10m
        ## generate np arrays
        shape = (bbox[1]-bbox[0],bbox[3]-bbox[2])
        # use the internal method to calculate exact pixel coordinates
        pixCoords = [self._utm_to_pp(refPixUtm,x,10) for x in ROI.coordinates_utm]
        # round these values
        rounded = [(round(x[0]),round(x[1])) for x in pixCoords]

        # get a new pixel string based on new georef of pixel
        mask = self._poly_to_mask(rounded,shape)
        return mask
    
    def _poly_to_mask(self,poly, shape):
        """
        Returns a boolean mask of a list of pixel coordinates
        """
        vertex_row_coords = [x[1] for x in poly]
        vertex_col_coords = [x[0] for x in poly]
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_row_coords, fill_col_coords] = True
        return mask
                
def shp_as_ROI_list(shp_file,coord_system='BNG',getRecords=True):
    """
        Returns a list of ROI objects for a shapefile - does this much faster
        than creating individual objects.
        
        Keyword Arguments:
        shp_file -- single .shp, .shx or .dbf file
        coord_system -- only supports BNG at present
        getRecords -- imports records as dictionary
    """
    if coord_system == 'BNG':
        sf = shapefile.Reader(shp_file)
        # get record entries and append to dictionary
        if getRecords==True:
            fields = sf.fields
            records = sf.records()
        roiList = []
        shapes = sf.shapes()
        for i in range(0,len(shapes)):
            roi = sentinelROI()
            BNGcoordinates = shapes[i].points
            # split coords, convert and append to self.lat and self.lon
            for pair in BNGcoordinates:
                lola = convert_lonlat(pair[0],pair[1])
                roi.lon.append(lola[0][0])
                roi.lat.append(lola[1][0])
            # run all init functions
            # make coordinates attribute
            roi._leaflet_coords()
            # extract the ROI lat/lon
            roi._convert_to_grid()
            # calculate bouding box coordinates
            roi._boxROI_utm()
            if getRecords == True:
                for j in range(1,len(sf.fields)):
                    roi.records[fields[j][0]] = records[i][j-1]
            
            
            # append to roiList
            roiList.append(roi)
        return(roiList)



def combine_ROIs(roiList):
    """ takes a list of ROIs and produces a square ROI object (bounding box)
    which includes all the ROIs in the list
    """
    lats = []
    lons = []
    for r in roiList:
        for c in r.coordinates:
            # if cLat < minLat, replace minLat
            lats.append(c[0])
            lons.append(c[1])
    minLat = min(lats)
    maxLat = max(lats)
    minLon = min(lons)
    maxLon = max(lons)
    newROI = sentinelROI()
    newROI.lat = [maxLat,maxLat,minLat,minLat,maxLat]
    newROI.lon = [minLon,maxLon,maxLon,minLon,minLon]
    # make coordinates attribute
    newROI._leaflet_coords()
    # extract the ROI lat/lon
    newROI._convert_to_grid()
    # calculate bouding box coordinates
    newROI._boxROI_utm()
    return(newROI)

def image_histogram_equalization(image, number_bins=256, cdf=None, bins=None):
    """ Image histogram equlisation"""
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    if cdf is None:
        image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape),cdf,bins



def is_geomatch(sentinelObject1,sentinelObject2):
    """
    Returns True if 2 objects are in the same 100km x 100km grid
    """
    if sentinelObject1.mgrsGridSquare == sentinelObject2.mgrsGridSquare:
        return True
    else:
        return False