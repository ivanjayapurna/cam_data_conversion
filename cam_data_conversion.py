# IMAGE TRANSFORM imports
from numpy import tan, deg2rad, array
import matplotlib.pyplot as plt
import cv2
import PIL.Image
from PIL.Image import fromarray
from image_utils import changeColorDepth
# DONKEY CAR imports
from donkeycar import Vehicle
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.datastore import Tub


#####################
##### FUNCTIONS #####
#####################

def reduce_angle(input_image, initial_angle=160, final_angle=90):
	"""
	Inputs:
		input_image		:	JPG image to be manipulated.
		initial_angle	:	Float. Viewing angle of input_image.
		final_angle		:	Float. Desired viewing angle of the output image.
	Returns:
		PNG input_image, modified to have a viewing angle of final_angle.
		If final_angle > initial_angle, don't modify. Keeping center of the
		output image the same as the input image.
	"""
	if (initial_angle > final_angle):
		# from dimensions of image calculate center of image
		(h, w) = input_image.shape[:2]
		(c_h, c_w) = (h / 2, w / 2)
		# calculate linear field of view (fov) ratio (init / final)
		fov_ratio = tan(deg2rad(initial_angle) / 2.0) / tan(deg2rad(final_angle) / 2.0)
		# use ratio to determine amount of image to crop
		(h2, w2) = (h/fov_ratio, w/fov_ratio)
		cropped = input_image[int(c_h - h2 / 2):int(c_h + h2 / 2), int(c_w - w2 / 2):int(c_w + w2 / 2)]
		#print('cropped image shape', cropped.shape)
		#cv2.imwrite('cropped_image.png', cropped)
		return cropped


def rgb2grey(input_image):
	"""
	Inputs:
		input_image	:	RGB JPG image to be manipulated.
	Returns:
		Greyscaled JPG image of rgb_image.
	"""
	grey = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
	#print('grey image shape', grey.shape)
	#cv2.imwrite('grey_image.png', grey)
	return grey


def reduce_bits_per_pixel(input_image, final_bits=8):
	"""
	Inputs:
		final_bits	:	Integer. The number of bits used to define an individual
						pixel in the output image, i.e. 2 bits => 2^2 = 4
						possible intensity values for any pixel.
	Returns:
		PNG input_image, modified where each pixel is represented by final_bits
		number of bits. Assume an unchanged dynamic range.
	"""
	im = PIL.Image.fromarray(input_image.astype('uint8'))
	reduce_bit = changeColorDepth(im, 4)
	#print('reduce_bit image success')
	#im.save('reduce_bit_image.png')
	return array(im)


def reduce_resolution(input_image, final_rows=224, final_cols=224):
	"""
	Inputs:
		input_image	:	JPG Image to be manipulated.
		final_rows	:	Integer. The number of rows in the output image.
		final_cols	:	Integer. The number of columns in the output image.
	Returns:
		PNG input_image, downsampled to have final_rows rows and final_cols
		columns. If final_rows or final_cols are larger than the number of
		rows or columns in input_image, respectively, pad with zeros.
	"""
	
	# calculate ratio of new image to old image
	# assume images will always be landscape oriented or square
	r = final_rows / input_image.shape[1]
	dim = (final_rows, int(input_image.shape[0] * r))

	# perform actual image resizing
	resized = cv2.resize(input_image, dim, interpolation = cv2.INTER_AREA)
	#print('resized image shape', resized.shape)
	#cv2.imwrite('resized_image.png',resized)
	return resized


def img2research(input_image, initial_angle=160, final_angle=90, final_rows=224, final_cols=224):
	"""
	Inputs:
		input_image		:	Input from RPi. Image to be manipulated.
		initial_angle	:	Float. Viewing angle of input_image.
		final_angle		:	Float. Desired viewing angle of the output image.
		final_bits		:	Integer. The number of bits used to define an
							individual pixel in the output image.
		final_rows		:	Integer. The number of rows in the output image.
		final_cols		:	Integer. The number of columns in the output image.
	Returns:
		PNG input_image, modified to have 
		a new viewing angle of final_angle, final_bits to represent each pixel
		intensity, and final_rows/cols for the new dimensions of the image.
	"""
	final = reduce_resolution(reduce_bits_per_pixel(rgb2grey(reduce_angle(input_image))))
	#print('final image shape', final.shape)
	#cv2.imwrite('final_image.png', final)
	return final


def vid2research(input_media, initial_angle=160, final_angle=90, final_rows=224, final_cols=224):
	# Create Video Capture object and read from input file
	# if camera input, pass in 0 instead of file name
	cap = cv2.VideoCapture(input_media)
	if (cap.isOpened() == False):
		print("Error opening video stream or file")

	# Default resolutions of the frame are obtained. The default resolutions are system dependent.
	# Convert the resolutions from float to integer.
	frame_width, frame_height = cap.get(3), cap.get(4)

	# calculating the expected output frame dimensions post research conversion
	# assume images will always be landscape oriented or square
	r = final_rows / frame_width
	dim = (final_rows, int(frame_height * r))
	frame_width, frame_height = dim[0], dim[1]

	# Define the codec and create VideoWriter object. Store output to file.
	out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('a','v','c','1'), 10, (frame_width, frame_height), isColor=False)

	# Read until video is completed
	while (cap.isOpened()):
		# capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			# convert frame image to research image
			frame = img2research(frame)
			
			# write frame into 'output.avi'
			out.write(frame)

			# display the resulting frame
			cv2.imshow('frame', frame)

			# press Q on keyboard to exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		else:
			break

	# when done, release video capture and write objects & close all frames
	cap.release()
	out.release()
	cv2.destroyAllWindows()


# STILL TO-DO:
def rpi2research(initial_angle, final_angle,
				final_bits, final_rows, final_cols):
	"""
	Inputs:
		initial_angle	:	Float. Viewing angle of input_image.
		final_angle		:	Float. Desired viewing angle of the output image.
		final_bits		:	Integer. The number of bits used to define an
							individual pixel in the output image.
		final_rows		:	Integer. The number of rows in the output image.
		final_cols		:	Integer. The number of columns in the output image.
	Returns:
		None.
	"""
	# Set up the connection to the Raspberry Pi
	# defines a vehicle to take and record pictures 10 times per second
	V = Vehicle()

	#add a camera part
	cam = PiCamera()
	V.add(cam, outputs=['image'], threaded=True)

	#add tub part to record images
	tub = Tub(path='~/mycar/get_started',
	          inputs=['image'],
	          types=['image_array'])
	V.add(tub, inputs=['image'])

	#start the drive loop at 10 Hz
	V.start(rate_hz=10)

	# Display the pretty input from the Raspberry Pi's camera
	### YOUR CODE HERE ###

	# Display the converted video. The conversion should be done in real-time.
	# Save the converted frames as you go if you'd like.
	### YOUR CODE HERE ###


##################
##### SCRIPT #####
##################

media_path = 'sample_pi_media'
media_name = 'sample_pi_camera_video'
vid2research(media_path + '/' + media_name + '.mp4')

# for image transformation testing
'''
media_name = 'RAW0029'
image = cv2.imread(media_path + '/' + media_name + '.jpg')
print('image shape', image.shape)
img2research(image)
'''




