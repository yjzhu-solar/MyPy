# export Jupyter notebook to HTML or slides with images embedded in Markdown cells
# codes from @imcomking on GitHub issue in https://github.com/jupyter/nbconvert/issues/699

import nbformat
import nbconvert
import os 
import argparse
import base64

def ipy2html(filename, output_dir,slides=False):
    with open(filename) as nb_file:
        nb_contents = nb_file.read()

    # Convert using the ordinary exporter
    notebook = nbformat.reads(nb_contents, as_version=4)
    if slides is True:
        outname = os.path.join(os.path.basename(filename).replace(".ipynb", ".slides.html"))   
        print("Converting to slides:", outname)    
        exporter = nbconvert.SlidesExporter()    
    else:
        outname = os.path.join(output_dir, os.path.basename(filename).replace(".ipynb", ".html"))
        print("Converting to HTML:", outname)
        exporter = nbconvert.HTMLExporter()
        
    body, res = exporter.from_notebook_node(notebook)

    # Create a list saving all image attachments to their base64 representations
    images = []
    for cell in notebook['cells']:
        if 'attachments' in cell:
            attachments = cell['attachments']
            for filename, attachment in attachments.items():
                for mime, base64_ in attachment.items():
                    images.append( [f'attachment:{filename}', f'data:{mime};base64,{base64_}'] )

        if 'img src=' in cell['source']:
            for line in cell['source'].split('\n'):
                if 'img src=' in line:
                    imagepath_ = line.split('img src="')[1].split('"')[0]
                    with open(imagepath_, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        images.append( [imagepath_, f'data:image/png;base64,{encoded_string}'] )

    # Fix up the HTML and write it to disk
    for itmes in images:
        src = itmes[0]
        base64_ = itmes[1]
        body = body.replace(f'src="{src}"', f'src="{base64_}"', 1)
        
    with open(outname, 'w') as output_file:
        output_file.write(body)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Jupyter notebook to HTML or slides with images embedded in Markdown cells')
    parser.add_argument('filename', type=str, help='input Jupyter notebook filename')
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('--slides', action='store_true', help='convert to slides')
    args = parser.parse_args()
    ipy2html(args.filename, args.output_dir, args.slides)
