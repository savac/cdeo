import xml
import glob

def remove_tags(text):
    return ''.join(xml.etree.ElementTree.fromstring(text).itertext())

def remove_tags_file(f_in, f_out):
    with open(f_in, 'r') as h_in, open(f_out, 'wb') as h_out:
        for row in h_in:
            clean = remove_tags(row)
            if clean != '':
                h_out.write(clean)

def remove_tags_folder(f_match='*.tml'):
    '''Removes the tags from all files that match the name in the f_match 
    and outputs the results to a new folder called 'output'.'''
    files_in = glob.glob(f_match)
    directory = os.path.dirname(files_in[0]) + '/output'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for f in files_in:
        file_out = directory + '/' + f
        remove_tags_file(f, file_out)
