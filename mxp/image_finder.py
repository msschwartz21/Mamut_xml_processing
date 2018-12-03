import os

XY = {
	0: [0,1,0,-80],
	20: [0,1,-10,-80], 
	75: [0,1,-70,-80], 
	125: [0,1,-95,-80],
	250: [0,1,-105,-80],
	375: [0,1,-110,-80],
	425: [0,1,-160,-80],
	500: [0,1,-90,-80]
}

YZ = {
	0: [2,1,35,-90],
	20: [2,1,35,-90], 
	75: [2,1,35,-90], 
	125: [2,1,35,-90],
	250: [2,1,35,-90],
	375: [2,1,35,-90],
	425: [2,1,35,-90],
	500: [2,1,35,-90]
}

XZ = {
	0: [0,2,0,40],
	20: [0,2,-10,40], 
	75: [0,2,-70,40], 
	125: [0,2,-95,40],
	250: [0,2,-105,40],
	375: [0,2,-110,40],
	425: [0,2,-160,40],
	500: [0,2,-90,40]
}

dshift = {'XY': XY, 'YZ': YZ, 'XZ': XZ}

def findImage(imdata,dim,time):
	'''Returns filepath for image of specified time and dimension
	Also returns list of xp,yp,xc,yc'''

	cwd = os.getcwd()

	os.chdir(imdata)

	lsdir = os.listdir()

	for d in lsdir:
		if dim in d:
			fld = d

	lsdir = os.listdir(imdata+'\\'+fld)
	for im in lsdir:
		if str(time) in im:
			image = im

	path = imdata + '\\' + fld + '\\' + image
	ddim = dshift[dim]
	shift = ddim[time]

	return(path,shift)