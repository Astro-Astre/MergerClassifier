# -*- coding: utf-8-*-
import csv
import os
from astropy.io import fits
def readCSV(path):
    ra = []
    dec = []
    with open(path) as f:
        header = csv.DictReader(f)
        for row in header:
            ra.append(row['ra'])
            dec.append(row['dec'])
        return ra,dec

class fitsDo:
    def __init__(self):
        pass
        

        
    # 打开filename的fits文件，后续使用直接调用hudl，主要用于drawPicture方法
    def openFits(self,filename):
        self.filename = filename
        self.hdul = fits.open(self.filename)


if __name__ == "__main__":
    # path = 'galaxy_redshift_radec.csv'
    # ra,dec=readCSV(path)
    pixscale = 0.012
    # for i in range(6780,len(ra)):
    #     Ra = float(ra[i])
    #     Dec = float(dec[i])
    #     os.system("wget 'https://www.legacysurvey.org/viewer/fits-cutout?ra=%f&dec=%f&layer=dr8&pixscale=%f&bands=g'\
    #         -O ./redshift_galaxy_data/g/%d.fits" %(Ra,Dec,pixscale,i))
    readFits_notmg = fitsDo()
    readFits_notmg.openFits(r'darg_mergers.fits')
    length_notmg = readFits_notmg.hdul[1].data.shape[0]
    ra = (readFits_notmg.hdul[1].data[0]['ra2']+readFits_notmg.hdul[1].data[0]['ra1'])/2
    dec = (readFits_notmg.hdul[1].data[0]['dec2']+readFits_notmg.hdul[1].data[0]['dec1'])/2
    for i in range(20):
        pixscale+=0.05
        os.system("wget 'https://www.legacysurvey.org/viewer/jpeg-cutout?ra=%f&dec=%f&layer=dr8&pixscale=%f&bands=grz'\
        -O ./raw_data/jpg/merger_zoom/1_%f.jpg" %(ra,dec,pixscale,pixscale))