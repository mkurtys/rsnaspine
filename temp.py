import SimpleITK as sitk

fn = './SPIDER_MR_Spine/images/15_t2.mha'
im = sitk.ReadImage(fn) 

# standardize the orientation
orient_filter = sitk.DICOMOrientImageFilter()
orient_filter.SetDesiredCoordinateOrientation('RPI')
reoriented_image = orient_filter.Execute(im)

#get array
image_data = sitk.GetArrayFromImage(im)
