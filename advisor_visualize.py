# Command line input arguments are:
# 1: name of the advisor project directory
# 2: number of most time-consuming loops to be shown (default 5)
# 3: number of MPI ranks on node (scaling factor for roofs) (default 1)
# 4: max memory level (4 = DRAM)
# 5: x axis maximum (log scale)
# 6: y axis maximum (lin scale)

# class myLoopObj():
#     def __init__(self,loop)
#     try:
#         self.ai = float(loop['self_arithmetic_intensity'])
#     except ValueError:
#         self.ai = 0.0
#     try:
#         self.gflops = float(loop['self_gflops'])
#     except ValueError:
#         self.gflops = 0.0
#     try:
#         self.time = float(loop['self_time'])
#     except ValueError:
#         self.time = 0.0
#     self.name = loop['loop_name']

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import datetime

matplotlib.rcParams.update({'font.size': 14})

t1 = datetime.datetime.now()

#doPrint = True
#doPlot = True

# Add advisor dir to sys.path
sys.path.append(os.environ['ADVISOR_XE_2018_DIR']+'/pythonapi/')

import advisor

#Parse input arguments

# 2: number of most time-consuming loops to be shown (default 5)
if len(sys.argv) > 2:
    numberOfLoops = int(sys.argv[2])
    if not 'ADVISOR_NOPRINT' in os.environ:
        print('Printing '+str(numberOfLoops)+' loops')
else:
    numberOfLoops = 5

# 3: number of MPI ranks on node (scaling factor for roofs) (default 1)
if len(sys.argv) > 3:
    scalingFactorForRoofs = int(sys.argv[3])
    if not 'ADVISOR_NOPRINT' in os.environ:
        print('Scaling down roofs by '+str(scalingFactorForRoofs))
else:
    scalingFactorForRoofs = 1

memlvls = ['L1','L2','LLC','DRAM']

# 4: max memory level (4 = DRAM) (default 4)
if len(sys.argv) > 4:
    maxMemLvl = int(sys.argv[4])
    if not 'ADVISOR_NOPRINT' in os.environ:
        print('Maximum memory level shown is '+memlvls[maxMemLvl - 1])
else:
    maxMemLvl = 4

# 6: y axis minimum (log scale)
if len(sys.argv) > 6:
    ymin = float(sys.argv[6]) / scalingFactorForRoofs
else:
    ymin = 1.0 / scalingFactorForRoofs

#Define plotting styles
colors = ['b','g','y','r']
styles = ['o','s','v','^','D',"<",">","*","h","H","+","1","2","3","4","8","p","d","|","_",".",","]

if numberOfLoops > len(styles):
    print('Unfortunately we only support plotting up to '+str(len(styles))+'loops.')
    print('Number of loops has been set to '+str(len(styles)))
    numberOfLoops = len(styles)

#Min and max marker sizes, marker size will be scaled linearly by self time between these values
maxMarkerSize = 14
minMarkerSize = 6

# Open advisor project in command line argument 1 and read data from it
# for plotting the roofs
project = advisor.open_project(sys.argv[1])
data = project.load(advisor.SURVEY)
tsdata = sorted(data.bottomup, key = lambda x: float(x['self_time']),reverse=True)

if hasattr(data,'memory_levels'):
    nMemLvls = data.memory_levels
else:
    nMemLvls = 1

if not 'ADVISOR_NOPRINT' in os.environ:
    print(str(nMemLvls)+' Memory levels found')    

nMemLvls = min(nMemLvls,maxMemLvl)

#Create a list of loops of interest
listOfLoopIds = list()
listOfSelfTimes = list()

if 'ADVISOR_LOOPS_ONLY' in os.environ:
    loopsOnly = True
else:
    loopsOnly = False

for loop in tsdata:
    
    #Filter out loops that don't have GFLOPS or AI data
    if(loop['self_gflops'] is not '' and loop['self_arithmetic_intensity'] is not ''):
        #Filter out functions        
        if((loopsOnly and not 'function' in loop['type'].lower()) or not loopsOnly):            
            listOfLoopIds.append(int(loop['loop_function_id']))
            listOfSelfTimes.append(float(loop['self_time']))

    if len(listOfLoopIds) >= numberOfLoops:
        break

maxSelfTime = max(listOfSelfTimes)

if 'ADVISOR_NOPRINT' not in os.environ:
    # Print debugging information from each data struct on screen
    f = '{:^45}|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}'
    print 140 * '-'
    print(f.format('Loop Name',
                   'Trip count',
                   'Self GFLOP',
                   'Self GFLOPs',
                   'Self Memory GB',
                   'Self AI',
                   'Self Time'))
            
    for level in range(nMemLvls):
        iloop = 0
        print 140 * '-'
        if hasattr(data,'set_memory_level'):
            data.set_memory_level(level)
            print('{:^15}'.format(memlvls[level]))
        print 140 * '-'
        for loop in data.bottomup:
            if iloop > numberOfLoops:
                break
            if int(loop['loop_function_id']) in listOfLoopIds:
                iloop += 1
                print(f.format(loop['loop_name'],
                               loop['average_trip_count'],
                               loop['self_gflop'],
                               loop['self_gflops'],
                               loop['self_memory_gb'],
                               loop['self_arithmetic_intensity'],
                               loop['self_time']))

#Create a figure and axis
fig = plt.figure(1,figsize=(10.67,6))
plt.clf()
ax = fig.gca()

nx = 10000
xmin = -2
if len(sys.argv) > 5:
    xmax = float(sys.argv[5])
else:
    def func(x):

        if x['self_arithmetic_intensity'] is not '' and int(x['loop_function_id']) in listOfLoopIds:
            return float(x['self_arithmetic_intensity'])
        else:
            return 0.0

    #Set memory level to DRAM for finding max AI
    if hasattr(data,'set_memory_level'):
        data.set_memory_level(maxMemLvl - 1)

    aisdata = sorted(data.bottomup, key = lambda x: func(x),reverse=True)
    try:
        xmax = 1+int(np.log10(float(aisdata[0]['self_arithmetic_intensity'])))
    except ValueError:
        xmax = 3

x = np.logspace(xmin,xmax,nx)

#Collect roofs for memory bandwidth and compute from the advisor data object
memroofs = list()
comproofs = list()

memroofs_single = list()
comproofs_single = list()
isKNL = False
for roof in data.roofs:
    if not 'single-thread' in roof.name:
        if 'bandwidth' in roof.name.lower():
            memroofs.append(roof)
        if 'peak' in roof.name.lower() and not 'SP' in roof.name:
            comproofs.append(roof)
    else:
        if 'bandwidth' in roof.name.lower():
            memroofs_single.append(roof)

            if 'mcdram' in roof.name.lower():
                isKNL = True

        if 'peak' in roof.name.lower() and not 'SP' in roof.name:
            comproofs_single.append(roof)

#Sort the roofs in order of descending bandwidth
smemroofs = sorted(memroofs, key = lambda x: x.bandwidth,reverse=True)
scomproofs = sorted(comproofs, key = lambda x: x.bandwidth,reverse=True)

#Modify roof labels for KNL
if isKNL:
    class myRoofObj():
        def __init__(self,name,bw):
            self.name = name
            self.bandwidth = bw

    smemroofs_knl = list()
    #Swap DRAM BW with MCDRAM, add DRAM BW manually
    bws = [smemroofs[0].bandwidth, smemroofs[1].bandwidth, 77.0e9, smemroofs[2].bandwidth]

    for roof,bw in zip(smemroofs,bws):
        smemroofs_knl.append(myRoofObj(roof.name,bw))

    #Sort again and overwrite smemroofs
    smemroofs = sorted(smemroofs_knl, key = lambda x: x.bandwidth, reverse = True)

#Find the indices where the bandwidth peak is equal to the compute peak
for roof in scomproofs:
    for ix in range(1,nx):
        if smemroofs[0].bandwidth * x[ix] >= roof.bandwidth and smemroofs[0].bandwidth * x[ix-1] < roof.bandwidth:
            setattr(roof,'x_elbow',x[ix-1])
            setattr(roof,'ix_elbow',ix-1)
            break

for roof in smemroofs:
    setattr(roof,'x_elbow',x[-2])
    setattr(roof,'ix_elbow',nx-1)
    for ix in range(1,nx):
        if (scomproofs[0].bandwidth <= roof.bandwidth * x[ix] and scomproofs[0].bandwidth > roof.bandwidth * x[ix-1]):
            setattr(roof,'x_elbow',x[ix-1])
            setattr(roof,'ix_elbow',ix-1)
            break        

#Plot the roofs on the axis ax
memroof_handles = list()
comproof_handles = list()

for roof in scomproofs:
    if roof.bandwidth > 0 and hasattr(roof,'ix_elbow'):
        y = np.ones(len(x)) * roof.bandwidth * 1e-9 / scalingFactorForRoofs
        comproof_handles.append(ax.plot(x[roof.ix_elbow:],y[roof.ix_elbow:],c='k',ls='-',lw='2'))

for color,roof in zip(colors,smemroofs):
    if roof.bandwidth > 0 and hasattr(roof,'ix_elbow'):
        y = x * roof.bandwidth * 1e-9 / scalingFactorForRoofs
        memroof_handles.append(ax.plot(x[:roof.ix_elbow+1],y[:roof.ix_elbow+1],c=color,ls='-',lw='2'))

#Plot the roofline data points on the axis
marker_handles = list()

#Loop over memory levels
for level in range(nMemLvls):

    #Set the memory level in the advisor data
    if hasattr(data,'set_memory_level'):
        data.set_memory_level(level)

    iloop = 0

    for loop in data.bottomup:

        #Look for loops whose indices have been stored
        if int(loop['loop_function_id']) in listOfLoopIds:

            name = loop['function_call_sites_and_loops']
            #name = loop['loop_name']

            #Clean up loop names for legend labels. First remove trailing $omp - stuff
            #iend = name.find('$')
            #iend = name.find(' at ')
            #if iend < 0:
            iend = None

            #iLineNum = loop['loop_name'].find('@')

            #Then remove leading underscores, they throw a warning and don't display on the legend
            istart = 0

            iunderscore = name.find('_')
            if iunderscore == 0:
                istart = 1

            istart = name.find(' at ') + 4

            #Only add legends on the first level to avoid duplicates
            if level == 0:
                l = name[istart:iend]
            else:
                l = None
            
            #Add a point to the plot if the AI and GFLOPS values exist
            try:
                markersize = float(loop['self_time']) / maxSelfTime * (maxMarkerSize - minMarkerSize) + minMarkerSize
                ax.plot(float(loop['self_arithmetic_intensity']),float(loop['self_gflops']),
                        c=colors[level],marker=styles[iloop],linestyle='None',ms=markersize)

                # Dummy markers for legend
                if level == 0:
                    marker_handles.append(ax.plot([],[],c = 'gray', marker=styles[iloop],
                                                  linestyle='None',ms=markersize,label=l)[0])

                iloop += 1
            except ValueError:
                #If the AI and GFLOPS values don't exist, do nothing and continue
                iloop += 1
                continue

        #To save time, stop when all loops have been found
        if iloop >= numberOfLoops:
            break

# drawing points using the same ax
ax.set_xscale('log')
ax.set_yscale('log')

ymax = 4000/scalingFactorForRoofs
#fix ylim for all plots
ax.set_ylim(ymin, ymax)

ax.set_xlim(10**xmin, 10**xmax)

#Set labels and legend
ax.set_xlabel('Arithmetic Intensity [FLOP/Byte]')
ax.set_ylabel('Performance [GFLOP/sec]')

#Set text labels on rooflines

#import label_lines as ll
#ll.labelLines(roof_handles)

ix = int(nx*0.02)
xlim = ax.get_xlim()
ylim = ax.get_ylim()

#Add text on the horizontal compute roofs
for roof in scomproofs:
    bw = float(roof.bandwidth) * 1e-9 / scalingFactorForRoofs
    ax.text(x[-ix],roof.bandwidth*1e-9/scalingFactorForRoofs,
            roof.name + ': ' + '{0:.1f}'.format(bw) + ' GFLOP/s',
            horizontalalignment='right',
            verticalalignment='bottom')

#Add tilted text on the bandwidth loops
for roof in smemroofs:
    dx = x[1]/x[0]
    bw = float(roof.bandwidth) * 1e-9 / scalingFactorForRoofs

    #The angle of a linear function on a log-log plot is always pi/sqrt(2). However, we have to take into
    #account the aspect ratio of the axes, and the aspect ratio of the figure. NOTE, if you resize the
    #figure it will screw up the angle of the text
    ang = 180 / np.pi *np.arctan(np.log10(xlim[1]/xlim[0]) / np.log10(ylim[1]/ylim[0]) 
                                 * fig.get_size_inches()[1]/fig.get_size_inches()[0] )

    #To raise the text above the line by roughly the same distance regardless of the angle I'm trying to use
    #a sin**2 - function. It doesn't work perfectly but is better than nothing.
    ax.text(x[ix],x[ix]*roof.bandwidth*1e-9/scalingFactorForRoofs*(1+0.5*np.sin(ang/180*np.pi)**2),
            roof.name + ': ' + '{0:.1f}'.format(bw) + ' GB/s',
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation=ang)

#Add legends. We make one for the symbols and one for the colors to avoid redundancy.

#First, the one for the symbols is just created from marker_handles
if len(marker_handles) > 1:
    leg1 = plt.legend(handles = marker_handles,loc='lower right', borderaxespad=0.,
                      bbox_to_anchor = (0.75,0.024),scatterpoints = 1)
    #leg1 = plt.legend(handles = marker_handles,loc=(0.2,0.015))

    #This is some matplotlib magic to keep the legend there after we create a second one
    ax.add_artist(leg1)

#Create dummy patches for the color legend
import matplotlib.patches as mpatches
patch_handles = list()
for roof,color in zip(smemroofs[:maxMemLvl],colors[:maxMemLvl]):
    i = roof.name.find(' Bandwidth')
    patch_handles.append(mpatches.Patch(color=color,label = roof.name[:i]))

#Create the second legend from patch_handles
leg2 = plt.legend(handles = patch_handles,loc=4)

fullfilename = sys.argv[1]
i = fullfilename.find('/')
filename = fullfilename
while i is not -1:
    i = filename.find('/')
    filename = filename[i+1:]

#Save figure into png and eps
plt.savefig(filename+'.png')
plt.savefig(filename+'.eps')

#Show the figure on screen
if 'ADVISOR_NODISP' not in os.environ:
    plt.show()

t2 = datetime.datetime.now()

print(filename+' Completed in '+str((60*t2.minute+t2.second+1e-6*t2.microsecond)-(
    60*t1.minute+t1.second+1e-6*t1.microsecond))+'s.')
