import numpy as np

# normalizeGreyScale will take a list of flattened image arrays and "normalize" them
# In reality, the scaling is from the pixel values from 255 - 0 into values from 1 - 0
def normalizeGreyScale(input):
    for img in input:
        for i in range(len(img)):
            img[i] /= 255

# prepareSquareImg will flatten the raw data collected in data_collect.ipynb -
# - and normalize the values   
def prepareSquareImg(input):
    input = input.reshape(-1, len(input[0])**2)
    input = input.astype(np.float32)
    normalizeGreyScale(input)
    return input

# Very annoying implementation, but getOneShot will give the indec for -
# the slice of the identity array (implemented in convertLabeltoOneshot) -
# - that represents what class it is meant to be given the label
# Important for the categorical cross entropy loss function
def getOneShot(label):
    value = label
    match label:
        case '36':
            value = 0        
        case '53_73':
            value = 1        
        case '4e':
            value = 2        
        case '41':
            value = 3        
        case '4c_6c':
            value = 4        
        case '4d_6d':
            value = 5        
        case '52':
            value = 6        
        case '71':
            value = 7        
        case '37':
            value = 8        
        case '49_69':
            value = 9        
        case '4f_6f':
            value = 10
        case '6e':
            value = 11
        case '38':
            value = 12
        case '39':
            value = 13
        case '50_70':
            value = 14
        case '35':
            value = 15
        case '32':
            value = 16
        case '31':
            value = 17
        case '30':
            value = 18
        case '59_79':
            value = 19
        case '62':
            value = 20
        case '65':
            value = 21
        case '72':
            value = 22
        case '33':
            value = 23
        case '43_63':
            value = 24
        case '4a_6a':
            value = 25
        case '34':
            value = 26
        case '55_75':
            value = 27
        case '5a_7a':
            value = 28
        case '54':
            value = 29
        case '68':
            value = 30
        case '42':
            value = 31
        case '64':
            value = 32
        case '61':
            value = 33
        case '74':
            value = 34
        case '57_77':
            value = 35
        case '47':
            value = 36
        case '51':
            value = 37
        case '46':
            value = 38
        case '4b_6b':
            value = 39
        case '58_78':
            value = 40
        case '45':
            value = 41
        case '44':
            value = 42
        case '56_76':
            value = 43
        case '67':
            value = 44
        case '48':
            value = 45
        case '66':
            value = 46
    return value

# Converts neural network output to ascii characters that will be covnerted using Ascii2Char()
def getLabel(label):
    value = label
    if value == 0:
        return '36'
    elif value == 1:
        return '53_73'
    elif value == 2:
        return '4e'
    elif value == 3:
        return '41'
    elif value == 4:
        return '4c_6c'
    elif value == 5:
        return '4d_6d'
    elif value == 6:
        return '52'
    elif value == 7:
        return '71'
    elif value == 8:
        return '37'
    elif value == 9:
        return '49_69'
    elif value == 10:
        return '4f_6f'
    elif value == 11:
        return '6e'
    elif value == 12:
        return '38'
    elif value == 13:
        return '39'
    elif value == 14:
        return '50_70'
    elif value == 15:
        return '35'
    elif value == 16:
        return '32'
    elif value == 17:
        return '31'
    elif value == 18:
        return '30'
    elif value == 19:
        return '59_79'
    elif value == 20:
        return '62'
    elif value == 21:
        return '65'
    elif value == 22:
        return '72'
    elif value == 23:
        return '33'
    elif value == 24:
        return '43_63'
    elif value == 25:
        return '4a_6a'
    elif value == 26:
        return '34'
    elif value == 27:
        return '55_75'
    elif value == 28:
        return '5a_7a'
    elif value == 29:
        return '54'
    elif value == 30:
        return '68'
    elif value == 31:
        return '42'
    elif value == 32:
        return '64'
    elif value == 33:
        return '61'
    elif value == 34:
        return '74'
    elif value == 35:
        return '57_77'
    elif value == 36:
        return '47'
    elif value == 37:
        return '51'
    elif value == 38:
        return '46'
    elif value == 39:
        return '4b_6b'
    elif value == 40:
        return '58_78'
    elif value == 41:
        return '45'
    elif value == 42:
        return '44'
    elif value == 43:
        return '56_76'
    elif value == 44:
        return '67'
    elif value == 45:
        return '48'
    elif value == 46:
        return '66'


# Converts complete lable list into target list with indeces representing slices of I
def convertLabel2Class(labels):
    target = []
    for label in labels:
        target.append(getOneShot(label))
    return target

# Converts labels to one shot hot encoding using getOneShot
def convertLabel2OneShot(labels):
    target = []
    I = np.eye(47)
    for label in labels:
        target.append(I[getOneShot(label)])
    return target

# Converts the by_merge ascii labels for each category to readable characters
def Ascii2Char(ascii):
    if(len(ascii) == 2):
        return bytes.fromhex(ascii).decode("ASCII")
    else:
        string = ascii.split('_', -1)
        char1 = bytes.fromhex(string[0]).decode("ASCII")
        char2 = bytes.fromhex(string[1]).decode("ASCII")
        return char1 + " or " + char2

# Converts from ascii label from by_merge to readable character
def Ascii2Char(ascii):
    if(len(ascii) == 2):
        return bytes.fromhex(ascii).decode("ASCII")
    else:
        string = ascii.split('_')
        char1 = bytes.fromhex(string[0]).decode("ASCII")
        char2 = bytes.fromhex(string[1]).decode("ASCII")        
        return char1 + " or " + char2