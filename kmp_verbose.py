import numpy as np
import matplotlib.pyplot as plt

class output_parser():

    def __init__(self,fn):

        fh = open(fn)

        lines = fh.readlines()

        self.thread  = dict()
        self.procset = dict()

        self.core = dict()
        self.proc = dict()
        
        for l in lines:

            if 'OMP: Info #242' in l:

                pid    = -1
                thread = -1
                proc   = -1
                
                for i,ele in enumerate(l.split()):
                    if 'pid' in ele:
                        pid = int(l.split()[i+1])
                    if 'thread' in ele:
                        thread = int(l.split()[i+1])
                    if 'proc' in ele:
                        procset = int(l.split()[i+2][1:-1])                

                if pid >= 0 and thread >= 0 and procset >= 0:

                    if not pid in self.thread.keys():
                        self.thread[ pid] = [thread]
                    else:                    
                        self.thread[ pid].append(thread)

                    if not pid in self.procset.keys():
                        self.procset[pid] = [procset]
                    else:
                        self.procset[pid].append(procset)

            if 'OMP: Info #171' in l:

                proc = -1
                core = -1
                thread = -1
                
                for i,ele in enumerate(l.split()):
                    if 'proc' in ele:
                        proc = int(l.split()[i+1])
                    if 'core' in ele:
                        core = int(l.split()[i+1])
                    if 'thread' in ele:
                        thread = int(l.split()[i+1])                

                if proc >= 0 and core >= 0 and thread >= 0:
                    if not thread in self.core.keys():
                        self.core[thread] = [core]
                    else:                    
                        self.core[thread].append(core)

                    if not thread in self.proc.keys():
                        self.proc[thread] = [proc]
                    else:
                        self.proc[thread].append(proc)

                        
    def plot_thread_bind(self,fignum=1):

        fig = plt.figure(fignum)
        plt.clf()
        for pid in self.thread.keys():
            plt.plot(self.thread[pid],self.procset[pid],'o',label='pid '+str(pid))
            plt.xlabel('thread #')
            plt.ylabel('proc set #')
            plt.legend(loc=9,ncol=4)

    def plot_proc_map(self,fignum=2):

        fig = plt.figure(fignum)
        plt.clf()
        for thread in self.proc.keys():
            plt.plot(self.core[thread],self.proc[thread],'o',label='thread '+str(thread))
            plt.xlabel('core #')
            plt.ylabel('OS proc #')
            plt.legend(loc=9,ncol=4)
            
