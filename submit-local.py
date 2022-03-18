import os, sys
import subprocess
import json
import uproot
import awkward as ak

from coffea import processor, util, hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from boostedhiggs import VBFProcessor
from boostedhiggs import DDTProcessor

from distributed import Client
from lpcjobqueue import LPCCondorCluster

from dask.distributed import performance_report
from dask_jobqueue import HTCondorCluster, SLURMCluster

year = sys.argv[1]

# get list of input files                                                                                                 
infiles = subprocess.getoutput("ls infiles/"+year+"*QCDt.json").split()

for this_file in infiles:

    index = this_file.split("_")[1].split(".json")[0]

    print(this_file, index)

    uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

    p = DDTProcessor(year=year,jet_arbitration='ddb',systematics=False)
    args = {'savemetrics':True, 'schema':NanoAODSchema}

    output = processor.run_uproot_job(
        this_file,
        treename="Events",
        processor_instance=p,
        executor=processor.iterative_executor, #processor.dask_executor,
        executor_args={
            "schema": NanoAODSchema,
        },
        #chunksize=100000,
        maxchunks=4,
    )

    outfile = 'outfiles/ddt_'+str(year)+'UL_dask_'+index+'.coffea'
    util.save(output, outfile)
    print("saved " + outfile)

