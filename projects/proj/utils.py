from scipy import misc
def process_input(img, width, height):
    out = img[:195, :] # get only the playing area of the image
    r, g, b = out[:,:,0], out[:,:,1], out[:,:,2]
    out = r * (299./1000.) + r * (587./1000.) + b * (114./1000.)
    out = misc.imresize(out, (width,height), interp="bilinear")
    return out