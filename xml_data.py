import os
import xml.etree.ElementTree as ET

data_text = []

for file_name in os.listdir('vn_data_xml'):
    try:
        root = ET.parse('vn_data_xml/' + file_name).getroot()
        for pair in root.findall('text/spair'):
            values = pair.findall('s')
            pair = []
            for value in values:
                pair.append(value.text)
            data_text.append(pair)
    except:
        print('error')

print(len(data_text))
for i in range(100):
    print('[%s] => [%s]' % (data_text[i][0], data_text[i][1]))