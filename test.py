import numpy as np
from PIL import Image

# Create a blank white canvas
x = np.full((1024, 1024, 3), 255, dtype=np.uint8)

# Steve's skin color
x[200:400, 400:600] = [158, 138, 107]  # skin color 

# Steve's hair (black)
x[200:250, 400:600] = [51, 39, 20]
x[250:275, 400:425] = [51, 39, 20]  
x[250:275, 575:600] = [51, 39, 20]

# Steve's eyes (white and blue)
x[300:330, 430:450] = [247, 247, 250]  # Left white part
x[300:330, 450:470] = [53, 53, 133]    # Left blue part
x[300:330, 550:570] = [247, 247, 250]  # Right white part
x[300:330, 530:550] = [53, 53, 133]    # Right blue part

# Steve's nose (brown) - narrower
x[330:355, 470:530] = [89, 67, 34]

# Steve's mouth (black)
x[375:400, 450:550] = [43, 30, 11]
x[355:375, 450:470] = [43, 30, 11]  # Left side of the mouth
x[355:375, 530:550] = [43, 30, 11]  # Right side of the mouth

# Steve's body (blue shirt)
x[400:700, 400:600] = [5, 221, 250]  # Blue shirt

# Steve's legs & shoes (darker blue & gray)
x[700:975, 400:600] = [53, 53, 133]    # Pants
x[950:1000, 400:600] = [158, 158, 158] # Shoes

# Steve's arms (same color as skin)
x[400:700, 300:400] = [158, 138, 107]  # Left arm
x[400:700, 600:700] = [158, 138, 107]  # Right arm

# Convert the array to an image and show it
image = Image.fromarray(x)
image.show()
