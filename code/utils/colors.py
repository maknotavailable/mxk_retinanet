import warnings


def label_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.
    Args
        label: The label to get the color for.
    Returns
        A list of three values representing a RGB color.
        If no color is defined for a certain label, the color green is returned and a warning is printed.
    """
    # specularity: pink, saturation: dark blue, artifact: light blue, blur: yellow, contrast: orange, bubbles: black, instrument: green
    colors = [
      [236  , 20   , 236] ,
      [12   , 16 , 134] ,
      [23 , 191  , 242]   ,
      [242 , 242  , 23]   ,
      [242 , 156   , 8]   ,
      [10 , 10  , 10]   ,
      [255   , 255 , 255]  ,
      [255 , 0   , 133] 
    ]
    
    if label < len(colors):
        return colors[label]
    else:
        warnings.warn('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)


