import warnings


def label_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.
    Args
        label: The label to get the color for.
    Returns
        A list of three values representing a RGB color.
        If no color is defined for a certain label, the color green is returned and a warning is printed.
    """
    # specularity: pink, saturation: dark blue, artifact: light blue, blur: yellow, contrast: orange, bubbles: black, instrument: white
    colors = [
      [236  , 20   , 236] , #pink
      [12   , 16 , 134] , #navyblue
      [23 , 191  , 242]   , #lightblue
      [242 , 242  , 23]   , #yellow
      [242 , 156   , 8]   , #orange
      [10 , 10  , 10]   , #black
      [255   , 255 , 255]  , #white
      [255 , 51   , 51] , #red
      [255 , 51   , 255] , #purple
    ]
    
    if label < len(colors):
        return colors[label]
    else:
        warnings.warn('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)