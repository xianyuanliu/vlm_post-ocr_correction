from tqdm import tqdm
import json
import os
import shutil
import xml.etree.ElementTree as ET

ROOT_PATH = '../../../../../mnt/xdrive/Data_Sets/Library'
BL1_PATH = 'Data1/BritishLibraryNewspapersPart1'
BL2_PATH = 'Data1/BritishLibraryNewspapersPart2'


# Builds local copy of BLN metadata, handling known issues
def build_metadata(collection):
    if collection == 'bl1':
        path = BL1_PATH
        file = 'bl1.json'
        print('Building', file)
    elif collection == 'bl2':
        path = BL2_PATH
        file = 'bl2_metadata.json'
        print('Building', file)
    else:
        return

    with open(os.path.join(ROOT_PATH, path, file), 'r') as f:
        metadata = json.load(f)

    records = []
    for r in tqdm(metadata['records']):
        if r['publication_id'] != r['data_location'].split('/')[3]:
            r['publication_id'] = r['data_location'].split('/')[3]
            r['publication_metadata'] = r['data_location'].split('/')[3] + '_PublicationMetadata.xml'
        records.append(r)
    print('Total number of records:', len(records))
    metadata['records'] = records

    with open(file, 'w') as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
    print('Built', file)


# Builds metadata for all BLN publications
def build_publication_metadata():
    file = 'publication.json'
    print('Building', file)

    name, collection, id_, language, city = [], [], [], [], []

    bl1_publications = '19thCenturyBritishLibraryNewspapers_Part1_02/XML/NEWSPAPERS'
    for p_id in tqdm(os.listdir(os.path.join(ROOT_PATH, BL1_PATH, bl1_publications))):
        tree = ET.parse(os.path.join(ROOT_PATH, BL1_PATH, bl1_publications, p_id, p_id + '_PublicationMetadata.xml'))
        name.append(tree.find('PublicationTitle').text.replace('&apos;', '\''))
        collection.append('bl1')
        id_.append(p_id)
        language.append(tree.find('VariantTitles/Language').text)
        city.append(tree.find('VariantTitles/City').text)

    bl2_publications = '19CenturyBritishLibraryNewspapersPartII_02/XML/NEWSPAPERS'
    for p_id in tqdm(os.listdir(os.path.join(ROOT_PATH, BL2_PATH, bl2_publications))):
        tree = ET.parse(os.path.join(ROOT_PATH, BL2_PATH, bl2_publications, p_id, p_id + '_PublicationMetadata.xml'))
        name.append(tree.find('PublicationTitle').text.replace('&apos;', '\''))
        collection.append('bl2')
        id_.append(p_id)
        language.append(tree.find('VariantTitles/Language').text)
        city.append(tree.find('VariantTitles/City').text)

    publications = []
    for i in range(len(name)):
        p = {'name': name[i],
             'collection': collection[i],
             'id': id_[i],
             'language': language[i],
             'city': city[i],
             }
        publications.append(p)
    publications.sort(key=lambda x: x['name'])

    with open(file, 'w') as f:
        json.dump(publications, f, indent=2)
    print('Built', file)


# Builds local copy of dataset for given publication
def build_dataset(publication):
    print('Building', publication)

    with open('publication.json', 'r') as f:
        publication_metadata = json.load(f)

    collection, path, id_ = '', '', ''
    for p in publication_metadata:
        if p['name'] == publication:
            if p['collection'] == 'bl1':
                collection = 'bl1'
                path = BL1_PATH
                with open('bl1.json', 'r') as f:
                    bl_metadata = json.load(f)
            elif p['collection'] == 'bl2':
                collection = 'bl2'
                path = BL2_PATH
                with open('bl2_metadata.json', 'r') as f:
                    bl_metadata = json.load(f)
            id_ = p['id']

    collection_dir = os.path.join('../', collection)
    if not os.path.isdir(collection_dir):
        os.mkdir(collection_dir)
    if not os.path.isdir(os.path.join(collection_dir, publication)):
        os.mkdir(os.path.join(collection_dir, publication))

    for r in tqdm(bl_metadata['records']):
        if r['publication_id'] == id_:
            article_id, asset_id, ocr_text = [], [], []

            issue = ET.parse(os.path.join(ROOT_PATH, path, r['data_location'], r['issue_file']))
            try:
                ocr = ET.parse(os.path.join(ROOT_PATH, path, r['data_location'], r['ocr_file']))
            except ET.ParseError:
                continue

            for a in issue.findall('page/article'):
                article_id.append(a.find('id').text)
                asset_id.append(a.find('assetID').text)

            for i, a in enumerate(ocr.findall('artInfo')):
                if article_id[i] == a.attrib['id']:
                    ocr_text.append(a.find('ocrText').text)

            articles = []
            for i in range(len(article_id)):
                a = {'article_id': article_id[i],
                     'asset_id': asset_id[i],
                     'ocr_text': ocr_text[i],
                     }
                articles.append(a)

            with open(os.path.join(collection_dir, publication, issue.find('metadataInfo/PSMID').text + '.json'), 'w') as f:
                json.dump(articles, f, indent=2)

    print('Built', publication)


if __name__ == '__main__':
    # Copy XML/IMG files from X: drive to local directory
    # sudo mount -t cifs //uosfstore.shefuniad.shef.ac.uk/shared /mnt/xdrive -o username=USERNAME,rw,file_mode=0700,dir_mode=0700,uid=LOCALUSER

    with open('data/metadata.json', 'r') as f:
        bln600 = json.load(f)

    xml = set()
    img = set()

    for doc in tqdm(bln600):
        src = os.path.join(ROOT_PATH, BL1_PATH, doc['xml'])
        dst = os.path.join('data/xml', os.path.basename(doc['xml']))
        if dst not in xml and os.path.exists(src):
            shutil.copy(src, dst)
            xml.add(dst)

        src = os.path.join(ROOT_PATH, BL1_PATH, doc['img'])
        dst = os.path.join('data/img', os.path.basename(doc['img']))
        if dst not in img and os.path.exists(src):
            shutil.copy(src, dst)
            img.add(dst)
