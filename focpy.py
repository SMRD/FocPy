# -*- coding: utf-8 -*-
"""
Created on Tue Aug 04 10:40:30 2015

@author: ISTI_EW
Focal Sphere determination for up-going (straight) rays
"""

import numpy as np, obspy, obspy.taup, pandas as pd, matplotlib.pyplot as plt, math,os, shutil
import mplstereonet, glob, sys
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import itertools
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import glob


def colored_bar(left, height, z=None, width=0.8, bottom=0, ax=None, maxgap=np.pi, **kwargs):
    if ax is None:
        ax = plt.gca()
    width = itertools.cycle(np.atleast_1d(width))
    bottom = itertools.cycle(np.atleast_1d(bottom))
    rects = []
    for x, y, h, w in zip(left, bottom, height, width):
        rects.append(Rectangle((x,y), w, h))
    #coll = PatchCollection(rects, array=z, **kwargs)
    coll = PatchCollection(rects, **kwargs)
    coll.set_array(z)
    coll.set_clim([0,maxgap])
    #coll.set_clim([0,maxgap])
    ax.add_collection(coll)
    ax.autoscale()
    return coll
    
def _initAzPlot():
    
    plt.figure(figsize=[12,6])
    gs = gridspec.GridSpec(2, 3,width_ratios=[10,.4,10],height_ratios=[1,10],wspace=.4)
    
    titleAx=plt.subplot(gs[0:3])
    ax1 = plt.subplot(gs[3], projection='polar')
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location('N')
    #ax1.set_yticks([])
    ax1.set_yticklabels('')
    ax1.set_xticklabels(['N','','E','','S','','W',''])
    #ax1 = plt.subplot(gs[2], projection='equal_angle_stereonet')
    #ax2 = plt.subplot(gs[3], projection='equal_angle_stereonet')
    ax2 = plt.subplot(gs[5], projection='polar')
    ax2.set_theta_direction(-1)
    ax2.set_theta_zero_location('N')
    ax2.set_ylim(0,np.pi/2)
    #ax2.set_yticks([])
    ax2.set_xticklabels(['N','','E','','S','','W',''])
    ax2.set_yticklabels('')
    axcmap=plt.subplot(gs[4])
    
    return ax1, ax2, axcmap, titleAx
    
def _readStations(stafile):
    stations=pd.read_csv(stafile)
    if not set(['NAME','LAT','LON','ELE']).issubset(stations.columns): 
        raise Exception ("'LAT','LON','ELE' columns not in station file")
    return stations

def _readEvents(eventFile):
    eves=pd.read_csv(eventFile)
    if not set(['NAME','LAT','LON','ELE']).issubset(eves.columns): 
        raise Exception ("'LAT','LON','ELE' columns not in station file")
    return eves

def _initStereoPlot():
    plt.figure(figsize=[12,6])
    gs = gridspec.GridSpec(3, 2,width_ratios=[10,10],height_ratios=[1,10,.5])
    
    titleAx=plt.subplot(gs[0:2])
    ax1 = plt.subplot(gs[2], projection='polar')
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location('N')
    ax1.set_ylim(0,np.pi/2.0)
    ax1.set_yticks(np.radians([30,60]))
    ax1.set_yticklabels(['30','60'])
    ax1.set_xticklabels(['N','','E','','S','','W',''])
    #ax1 = plt.subplot(gs[2], projection='equal_angle_stereonet')
    #ax2 = plt.subplot(gs[3], projection='equal_angle_stereonet')
    ax2 = plt.subplot(gs[3], projection='polar')
    ax2.set_theta_direction(-1)
    ax2.set_theta_zero_location('N')
    ax2.set_ylim(0,np.pi/2)
    ax2.set_yticks(np.radians([30,60]))
    ax2.set_yticklabels(['30','60'])
    ax2.set_xticklabels(['N','','E','','S','','W',''])
    ax1cmap=plt.subplot(gs[4])
    ax2cmap=plt.subplot(gs[5])
    
    return ax1, ax2, ax1cmap, ax2cmap, titleAx

def _plotPoles(ax,df,cm,cax):
    #deb([ax,df,cax])
    distRange=[df.MapDistance.min(),df.MapDistance.max()]
    norm = mpl.colors.Normalize(vmin=distRange[0], vmax=distRange[1])
    for num,row in df.iterrows():
        if row.Proposed:
            ax.scatter(np.radians(row.Azimuth),np.radians(abs(row.TakeOff)),c=cm(row.MapDistance/max(distRange)),marker=(5,1),s=150)
        else:
            ax.scatter(np.radians(row.Azimuth),np.radians(abs(row.TakeOff)),c=cm(row.MapDistance/max(distRange)),s=75)        
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cm, norm=norm, spacing='proportional',orientation='horizontal')
        cb.set_label('Distance (km)')
            
def _createModel(vmod):
    if os.path.exists(vmod.split('.')[0]+'.npz'):
         model=obspy.taup.TauPyModel(vmod.split('.')[0]+'.npz')
    else:
        cwd=os.getcwd()
        obspy.taup.taup_create.build_taup_model(vmod,'TEMP')
        files=glob.glob(os.path.join('TEMP','*'))
        for fi in files:
            shutil.copy2(fi,os.path.join(cwd,os.path.basename(fi)))
        model=obspy.taup.TauPyModel(fi)
        shutil.rmtree(os.path.join(cwd,'TEMP'))
        return model
        
def getGeometry(lat1,lon1,lat2,lon2,r=6371):
    """
    get arc distance, km distance, and bearings assuming earth is a perfect sphere with radius of r (in km)
    """
    distm,az,backaz=obspy.core.util.geodetics.gps2DistAzimuth(lat1,lon1,lat2,lon2)
    distd=((distm/1000.)/r)*180/np.pi
    return distm, distd, az, backaz

def makeFocPy(stations='Stations.csv',events='Events.csv',vmod='VelocityModel.npz',phases=['p','P']):
    """
    Initialize FocPy class
    """
    fpy=FocPy(stations,events,vmod,phases)
    return fpy


    
def deb(ddl):
    global de
    de=ddl
    sys.exit(1)
    
class FocPy (object):
    """
    Object for 
    """
    def __init__(self,stations,events,vmod,phases):
        self.Stations=_readStations(stations)
        self.Events=_readEvents(events)
        
        #check that all files are there
        if not all ([os.path.exists(x) for x in [stations,events]]): raise Exception ('Not all required files exist')
        
        # load velocity model        
        if vmod.split('.')[1]=='npz':
            self.Model=obspy.taup.TauPyModel(vmod)
        elif vmod.split('.')[1]=='tvel':
            self.Model=_createModel(vmod)
        
        # Split into upgoing and downgoing ray paths 
        self.stadfs=[] #list to store pltdfs in for each station
        for evenum,everow in self.Events.iterrows(): #loop each event
            pltdf=pd.DataFrame(index=self.Stations.NAME,columns=['Phase','TakeOff','Incident','TT','RayParam','UpGoing','MapDistance','Depth','Azimuth','Proposed']) #container dataframe for plot parameters
            for stanum,starow in self.Stations.iterrows(): #loop each station
                #deb([starow,everow])
                distm, distd, az, backaz=getGeometry(starow.LAT,starow.LON,everow.LAT,everow.LON)
                azimuth=backaz
                dep=-1*(everow.ELE-starow.ELE)/1000. #get depth of each event from moving datum
                rever=0 #station and event are switched to trick taup into calculating for source below receiver
                if dep>0:
                    phas=self.Model.get_travel_times(dep,distd,phase_list=['p','P'])
                else:
                    dep=dep*-1
                    rever=1
                    phas=self.Model.get_travel_times(dep,distd,phase_list=['p','P'])
                if len(phas)<1:
                    continue
                phaDF=self._parsePhases(phas,distm,dep,azimuth,rever,starow)
                if azimuth<180:
                    pass
                pltdf.loc[starow.NAME]=phaDF.iloc[0]
            pltdf=pltdf[[not np.isnan(x) for x in pltdf.UpGoing]] #make sure no stations that had no P or p phases survives
            pltdf['EventName']=everow.NAME
            self.stadfs.append(pltdf)
    
    def plotFocalSpheres(self,maxDistance=None,minDistance=None,saveFigs=False):
        
        for pltdf in self.stadfs:
            ax1,ax2,ax1cmap,ax2cmap,titleAx=_initStereoPlot() #get axes
            pltdfUp=pltdf[pltdf.UpGoing] #seperate upgoing from down-going
            pltdfDown=pltdf[~pltdf.UpGoing]
            
            #get color maps and plot
            cmup = plt.cm.get_cmap('OrRd') #filter and plot ups
            if maxDistance:
                pltdfUp=pltdfUp[pltdfUp.MapDistance>minDistance]
            if minDistance:
                pltdfUp=pltdfUp[pltdfUp.MapDistance<maxDistance]
            _plotPoles(ax1,pltdfUp,cmup,ax1cmap)
            
            cmdown = plt.cm.get_cmap('GnBu') #filter and plot downs
            if maxDistance:
                pltdfDown=pltdfDown[pltdfDown.MapDistance>minDistance]
            if minDistance:
                pltdfDown=pltdfDown[pltdfDown.MapDistance<maxDistance]
            _plotPoles(ax2,pltdfDown,cmdown,ax2cmap)
            ax1.set_title('Up-Going Rays',y=1.1)
            ax2.set_title('Down-Going Rays',y=1.1)
            
            #set Title
            titleAx.set_frame_on(False)
            titleAx.set_xticks([])
            titleAx.set_yticks([])
            titleAx.set_title('Event : %s' % pltdf.EventName.iloc[0], fontsize=20)
            
            if saveFigs:
                plt.savefig(pltdf.iloc[0].EventName+'_FocalSpheres.pdf')
            else:
                plt.show()
    
    def _filterStationEvents(self,maxStationDistance=None,minStationDistance=None,**kwargs):
        """
        impose all the requirements on both the stations and the events then return stations and events
        """
        stadf=self.Stations
        evedf=self.Events
        
        ## find stations that meet distance requirements

        if maxStationDistance or minStationDistance:
            stadf['MinDist']=0
            stadf['MaxDist']=np.inf
            for num,row in stadf.iterrows():
               stadf.loc[num,'MinDist']=min([obspy.core.util.geodetics.gps2DistAzimuth(row.LAT,row.LON,ro.LAT,ro.LON)[0]/1000. for nu,ro in self.Events.iterrows()])
               stadf.loc[num,'MaxDist']=max([obspy.core.util.geodetics.gps2DistAzimuth(row.LAT,row.LON,ro.LAT,ro.LON)[0]/1000. for nu,ro in self.Events.iterrows()])
            if maxStationDistance: stadf=stadf[stadf.MinDist<maxStationDistance]
            if minStationDistance: stadf=stadf[stadf.MaxDist>minStationDistance]
        #print len(stadf)
        return stadf,evedf
    
    def plotAzGaps(self,maxStationDistance=None, minStationDistance=None,saveFigs=False):
        """
        Plot rose diagrams of each of the events with azmutial gaps displayed
        
        Parameters
        ----------
        maxStationDistance : float, int, or None
            The max distance form an event for a station to plot
        minStationDistance : float, int, or None
            The min distance from any event for a station to plot
        """
        
        #stadf,evedf=self._filterStationEvents(maxStationDistance=maxStationDistance,minStationDistance=minStationDistance)
        
        #staCurs=stadf[[not x for x in stadf.Proposed]]
         
        
        for evenum,stadf in enumerate(self.stadfs):
            plt.figure()
            ax1, ax2, axcmap, titleAx = _initAzPlot() 
            everow=self.Events.loc[evenum]
            if minStationDistance:
                stadf=stadf[stadf.MapDistance>minStationDistance]
            if maxStationDistance:
                stadf=stadf[stadf.MapDistance<maxStationDistance]
            stadfc=stadf[[ not x for x in stadf.Proposed]]
            fig = plt.figure()
            
            ax1.set_title('Current Network',y=1.08)
            ax2.set_title('Proposed Network',y=1.08)
            x1,y1,z1=self._getAzArray(stadf)
            x2,y2,z2=self._getAzArray(stadfc)
            
            cmap = plt.get_cmap('Spectral')
            maxz=np.max([max(z1),max(z2)])
            coll1 = colored_bar(x1, y1, z1, ax=ax2, width=y1, cmap=cmap, maxgap=maxz, alpha=.8)
            coll2 = colored_bar(x2, y2, z2, ax=ax1, width=y2, cmap=cmap, maxgap=maxz, alpha=.8)
            ax1.set_ylim([0,np.radians(maxz)])
            ax2.set_ylim([0,np.radians(maxz)])
            
            titleAx.set_frame_on(False)
            titleAx.set_xticks([])
            titleAx.set_yticks([])
            titleAx.set_title('Event : %s' % everow.NAME, fontsize=20)
            
            norm = mpl.colors.Normalize(vmin=0, vmax=maxz)
            cb = mpl.colorbar.ColorbarBase(axcmap, cmap=cmap, norm=norm, spacing='proportional',orientation='vertical')
            cb.set_label('Azmuthial Gap (deg)')
            
            if saveFigs:
                plt.savefig(stadf.iloc[0].EventName+'_AzGaps.pdf')
            else:
                plt.show()
            
    def _getAzArray(self,stadf):
        ar=list(stadf.Azimuth.values)
        ar.sort()
        ar.append(ar[0]+360)
        dif=np.diff(ar)
        x = np.radians(ar)
        y = np.radians(dif)
        z = dif
        return x,y,z
    
    def plotMap(self,maxStationDistance=None, minStationDistance=None, saveFigs=False):
        """
        Make plot of events (as beachball) and stations.
        
        Parameters
        ----------
        maxDistance : float or int
            The maximum distance from an event for stations to plot
        """
        ##Try to import basemap         
        try: 
            from  mpl_toolkits.basemap import Basemap
        except ImportError:
            raise ImportError('mpl_toolskits does not have basemap, plotting cannot be perfromed')
        
        plt.figure()
        stadf,evedf=self._filterStationEvents(maxStationDistance=maxStationDistance, minStationDistance=minStationDistance)
        stadfcur=stadf[[not x for x in stadf.Proposed]]
        stadfpro=stadf[[bool(x) for x in stadf.Proposed]]
        ## set up map basics
        latmin=min([evedf.LAT.min(),stadf.LAT.min()])
        lonmin=min([evedf.LON.min(),stadf.LON.min()])
        latmax=max([evedf.LAT.max(),stadf.LAT.max()])
        lonmax=max([evedf.LON.max(),stadf.LON.max()])
        
        latbuff=abs((latmax-latmin)*0.1) #create buffers so there is a slight border with no events around map
        lonbuff=abs((lonmax-lonmin)*0.1)
        totalxdist=obspy.core.util.geodetics.gps2DistAzimuth(latmin,lonmin,latmin,lonmax)[0]/1000. #get the total x distance of plot in km
        #totalydist=obspy.core.util.geodetics.gps2DistAzimuth(latmin,lonmin,latmax,lonmin)[0]/1000.
        
        
        emap=Basemap(projection='merc', lat_0 = np.mean([latmin,latmax]), lon_0 = np.mean([lonmin,lonmax]),
                     resolution = 'h', area_thresh = 0.1,
                     llcrnrlon=lonmin-lonbuff, llcrnrlat=latmin-latbuff,
                     urcrnrlon=lonmax+lonbuff, urcrnrlat=latmax+latbuff)
        emap.drawmapscale(lonmin, latmin, lonmin, latmin, totalxdist/4.5)
        #emap.drawrivers()
        
        xmax,xmin,ymax,ymin=emap.xmax,emap.xmin,emap.ymax,emap.ymin
        #horrange=max((xmax-xmin),(ymax-ymin)) #horizontal range
        
        x,y=emap(stadfcur.LON.values,stadfcur.LAT.values)
        emap.plot(x,y,'^',color='k',ms=6.0)
        
        x,y=emap(stadfpro.LON.values,stadfpro.LAT.values)
        emap.plot(x,y,'^',color='b',ms=6.0)
        
        x,y=emap(evedf.LON.values,evedf.LAT.values)
        emap.plot(x,y,'.',color='r',ms=7.0)
        emap.drawstates()
        
        for evenum,everow in evedf.iterrows():
            x,y=emap(everow.LON,everow.LAT)
            txt=everow.NAME
            plt.annotate(txt,xy=(x,y),xycoords='data')    
        
        latdi,londi=[abs(latmax-latmin),abs(lonmax-lonmin)] #get maximum degree distance for setting scalable ticks   
        maxdeg=max(latdi,londi)
        parallels = np.arange(0.,80,maxdeg/4)
        emap.drawparallels(parallels,labels=[1,0,0,1],linewidth=0.0)
        meridians = np.arange(10.,360.,maxdeg/4)
        emap.drawmeridians(meridians,labels=[1,0,0,1],rotation=90,linewidth=0.0)     
        
        if saveFigs:
            plt.savefig('Map.pdf')
        else:
            plt.show()
    
    
    def _parsePhases(self,phas,distm,dep,az,rever, starow):
        """
        Parse the phases object and return first arrivial
        """
        df=pd.DataFrame(index=range(len(phas)),columns=['Phase','TakeOff','Incident','TT','RayParam','MapDistance','Depth','Azimuth','Proposed']) 
        df['UpGoing']=False
        for num,pha in enumerate(phas):
            df.loc[num,'Phase']=pha.phase.name
            df.loc[num,'TakeOff']=pha.takeoff_angle
            df.loc[num,'Incident']=pha.incident_angle
            df.loc[num,'TT']=pha.time
            df.loc[num,'RayParam']=pha.ray_param
            df.loc[num,'MapDistance']=distm/1000.
            df.loc[num,'Depth']=dep
            df.loc[num,'Azimuth']=az
            df.loc[num,'Proposed']=starow.Proposed
            if rever: #switch takeoff angles and 
                df.loc[num,'Depth']=-dep
                df.loc[num,'TakeOff']=-pha.incident_angle
                df.loc[num,'Incident']=pha.takeoff_angle
        df=df[df.TT==df.TT.min()]
        #if name is 'p' or if takeoff is nearly vertical they are probably upgoing rays
        if df.iloc[0].Phase=='p' or abs(df.iloc[0].TakeOff)>89.: #adjust takeoff angle assuming straight ray paths
            df['UpGoing']=True
            if abs(df.iloc[0].TakeOff)>89.:
                distkm=distm/1000.0 #get distance in km to match dep
                df.iloc[0].TakeOff=np.arctan(distkm/dep)*180/np.pi
                df.TakeOff=np.arctan(distkm/dep)*180/np.pi
                
        return df
    
if __name__=='__main__':
    foc=makeFocPy(vmod="LF.npz")
    cwd=os.getcwd()
    for maxdist in [50]:
        os.chdir(cwd)
        newdir=os.path.join(cwd,str(maxdist)+'km')
        if not os.path.exists(newdir): os.makedirs(newdir)
        os.chdir(newdir)
        
        try: foc.plotAzGaps(maxStationDistance=maxdist,saveFigs=True) 
        except: pass
        
        try: foc.plotFocalSpheres(maxDistance=maxdist,saveFigs=True)
        except: pass
    
        try: foc.plotMap(maxStationDistance=maxdist,saveFigs=True)
        except: pass
    
    #foc.plotAzGaps()
    #foc.plotFocalSpheres()
    #foc.plotMap(maxStationDistance=30)
    #foc.plotAzGaps(maxStationDistance=30)
            
        
