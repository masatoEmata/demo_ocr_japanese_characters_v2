import numpy

def parse_label_digits(record):
    data = record.split(',')
    label = data[0]
    digits = data[1:]
    return (label, digits)

def scaled_input(digits):
    max = 255.0
    min = 0.01
    picture = numpy.asfarray(digits)
    return (picture / max * 0.99) + min