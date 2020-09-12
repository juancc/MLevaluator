"""
Image sub area cropper

JCA
Vaico
"""

def crop_rect(im, area, weights, conditions = None):
    """Calculated sub area of a area based on the weights of each dimension
        * area -> (x1, y1, x2, y2)
         BoundBox(xmin, ymin, xmax, ymax)

        * weights for: (xi, yi, w, h)
        * Weights of initial coords (xi,yi) are regarding the bb width:
            0 -> xi
            1 -> xi+ width
        * max_dim -> (max_x,max_y) | Max dimensions of the image

    Return -> _x1, _y1, _x2, _y2
    CONDITIONS
        center_x: center x direction
        center_y: center y direction
        square_w: square to width (after weighted)
        square_h: square to height (after weighted)
    """
    conditions = [] if not conditions else conditions

    max_dim = (im.shape[1], im.shape[0])
    # All area inside max dimensions
    # 0 < area < max_dim
    x1 = max(area['xmin'],0)
    y1 = max(area['ymin'],0) 
    x2 = max(area['xmax'],0) 
    y2 = max(area['ymax'],0)

    x1 = min(x1,max_dim[0])
    x2 = min(x2,max_dim[0]) 
    y1 = min(y1,max_dim[1]) 
    y2 = min(y2,max_dim[1])

    w1, w2, w3, w4 = weights
    w1 = float(w1)
    w2 = float(w2)
    w3 = float(w3)
    w4 = float(w4)

    W = abs(x2-x1)
    H = abs(y2-y1)

    _x1 = x1 + w1 * W
    _y1 = y1 + w2 * H

    if "square_h" in conditions:
        H_f = H*w4
        W_f = H_f
    if "square_w" in conditions:
        W_f = W*w3
        H_f = W_f
    else:
        W_f = W*w3
        H_f = H*w4
    _x2 = _x1 + W_f
    _y2 = _y1 + H_f

    if 'center_x' in conditions:
        offset = (W_f-W)/2
        _x1 = _x1 - offset
        _x2 = _x2 - offset
    if 'center_y' in conditions:
        offset = (H_f-H)/2
        _y1 = _y1 - offset
        _y2 = _y2 - offset

    # TESTING
    # Clip negative values
    _x1 = clip(_x1)
    _x2 = clip(_x2)
    _y1 = clip(_y1)
    _y2 = clip(_y2)
    # Check if 1 is smaller than 2 otherwise swap values
    if(_x1>_x2):
        _x1,_x2 =_x2,_x1
    if(_y1>_y2):
        _y1,_y2 =_y2,_y1
    # Check if crop is out of boundaries
    _x2 = max_dim[0] if (_x2>max_dim[0]) else _x2
    _y2 = max_dim[1] if (_y2>max_dim[1]) else _y2
     # if side of the crop is 0
    if(int(_x1-_x2)==0):
        if(_x2+1 > max_dim[0]):
            _x1 = _x1-1
        else:
            _x2 = _x2+1
    if(int(_y1-_y2)==0):
        if(_y2+1 > max_dim[1]):
            _y1 = _y1-1
        else:
            _y2 = _y2+1
            
    im_crop = im[int(_y1): int(_y2),  int(_x1) : int(_x2)]

    return  im_crop

def clip(n):
    return n if n>0 else 0
