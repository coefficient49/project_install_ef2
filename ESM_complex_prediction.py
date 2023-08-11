from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os, dnaio, argparse
import numpy as np
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
from scipy.special import softmax
from tqdm.auto import tqdm
from pathlib import Path
import pickle

def parse_output(output):
  pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
  plddt = output["plddt"][0,:,1]
  
  bins = np.append(0,np.linspace(2.3125,21.6875,63))
  sm_contacts = softmax(output["distogram_logits"],-1)[0]
  sm_contacts = sm_contacts[...,bins<8].sum(-1)
  xyz = output["positions"][-1,0,:,1]
  mask = output["atom37_atom_exists"][0,:,1] == 1
  o = {"pae":pae[mask,:][:,mask],
       "plddt":plddt[mask],
       "sm_contacts":sm_contacts[mask,:][:,mask],
       "xyz":xyz[mask]}
  if "contacts" in output["lm_output"]:
    lm_contacts = output["lm_output"]["contacts"].astype(float)[0]
    o["lm_contacts"] = lm_contacts[mask,:][:,mask]
  return o

def get_hash(x): 
  return hashlib.sha1(x.encode()).hexdigest()

alphabet_list = list(ascii_uppercase+ascii_lowercase)



jobname = "test" #@param {type:"string"}
jobname = re.sub(r'\W+', '', jobname)[:50]

def fix_sequence(jobname = "test",
                 sequence = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE",
                num_recycles = 6, #@param ["0", "1", "2", "3", "6", "12"] {type:"raw"}
                get_LM_contacts = False, #@param {type:"boolean"}):  #@param {type:"string"}
                #for homo_oligomer
                copies = 1, #@param {type:"integer"} 
                chain_linker = 25, #@param {type:"number"}
                samples = 8, #@param ["None", "1", "4", "8", "16", "32", "64"] {type:"raw"}
                masking_rate = 0.15, #@param {type:"number"}
                stochastic_mode = "LM", #@param ["LM", "LM_SM", "SM"]
                outdir="."
                ):

    sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
    sequence = re.sub(":+",":",sequence)
    sequence = re.sub("^[:]+","",sequence)
    sequence = re.sub("[:]+$","",sequence)

    if copies == "" or copies <= 0: 
      copies = 1
    sequence = ":".join([sequence] * copies)

       
    # ID = jobname+"_"+get_hash(sequence)[:5]
    ID = jobname + "_unrelaxed_rank_"#001_alphafold2_multimer_v3_model_4_seed_000.pdb
    seqs = sequence.split(":")
    lengths = [len(s) for s in seqs]
    length = sum(lengths)
    print("length",length)



    u_seqs = list(set(seqs))
    if len(seqs) == 1: mode = "mono"
    elif len(u_seqs) == 1: mode = "homo"
    else: mode = "hetero"

    if "model" not in dir():
        import torch
        model = torch.load("esmfold.model")
        # model = model.half()
        model.esm = model.esm.half()
        torch.backends.cuda.matmul.allow_tf32 = True
        model.cuda().requires_grad_(False)
    
        # optimized for Tesla T4
        if length >= 1200:
          model.trunk.set_chunk_size(8)
        elif 700 >= length > 1200:
          model.trunk.set_chunk_size(64) 
        else:
          model.trunk.set_chunk_size(128)
    
        best_pdb_str = None
        best_ptm = 0
        best_output = None
        traj = []
    
        num_samples = 1 if samples is None else samples
        for seed in tqdm(range(num_samples)):
            torch.cuda.empty_cache()
            if samples is None:
                seed = "default"
                mask_rate = 0.0
                model.train(False)
            else:
                torch.manual_seed(seed)
                mask_rate = masking_rate if "LM" in stochastic_mode else 0.0
                model.train("SM" in stochastic_mode)
    
            output = model.infer(sequence,
                                num_recycles=num_recycles,
                                chain_linker="X"*chain_linker,
                                residue_index_offset=512,
                                mask_rate=mask_rate,
                                return_contacts=get_LM_contacts)
            
            pdb_str = model.output_to_pdb(output)[0]
            output = tree_map(lambda x: x.cpu().numpy(), output)
            ptm = output["ptm"][0]
            plddt = output["plddt"][0,:,1].mean()
            traj.append(parse_output(output))
            print(f'{seed} ptm: {ptm:.3f} plddt: {plddt:.1f}')
            if ptm > best_ptm:
                best_pdb_str = pdb_str
                best_ptm = ptm
                best_output = output
            os.system(f"mkdir -p {jobname}")
            #ID = jobname + "_unrelaxed_rank_
            if samples is None:
                pdb_filename = f"{outdir}/{ID}_ptm{ptm:.3f}_r{num_recycles}_{seed}.pdb"
            else:
                pdb_filename = f"{outdir}/{ID}_ptm{ptm:.3f}_r{num_recycles}_seed{seed}_{stochastic_mode}_m{masking_rate:.2f}.pdb"
            
            with open(pdb_filename,"w") as out:
                out.write(pdb_str)
            with open(pdb_filename.replace(".pdb",".pickle"),"wb") as out:
                pickle.dump(output)

def get_args():

  parser = argparse.ArgumentParser(
      prog="ESMcomplex"
  )
  parser.add_argument(
      "-f",
      "--fasta",
      required=True,
      dest="fasta"
  )

  parser.add_argument(
      "-n",
      "--num_recycles",
      dest="num_recycles",
      default=6,
      type=int
  )

  parser.add_argument(
      "-LM",
      "--get_LM_contacts",
      dest="get_LM_contacts",
      default=False,
      type=bool
  )

  parser.add_argument(
      "-c",
      "--chain_linker",
      dest="chain_linker",
      default=24,
      type=int
  )

  parser.add_argument(
      "-s",
      "--samples",
      dest="samples",
      default=8,
      choices=[1,4,8,16,32,64],
      type=int
  )

  parser.add_argument(
      "-m",
      "--masking_rate",
      dest="masking_rate",
      default=0.15,
      type=float
  )

  parser.add_argument(
      "-SM",
      "--stochastic_mode",
      dest="stochastic_mode",
      default="LM",
      choices=["LM", "LM_SM", "SM"],
      type=str
  )



  parser.add_argument(
      "-o",
      "--out",
      default="default",
      dest="out"
  )

  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = get_args()

  fasta = args.fasta
  jobname = str(Path(fasta).with_suffix("").name)
  
  outdir = args.out
  
  if outdir == 'default':
     fasta_p = Path(fasta).with_suffix("")
  else:
     fasta_p = Path(outdir).with_suffix("")

  if not fasta_p.is_dir():
    fasta_p.mkdir(parents=True)
    print(f"making folder {fasta_p}")
  outdir = str(fasta_p)

  seq = [x.sequence for x in dnaio.FastaReader(fasta)]
  seq = ":".join(seq)


  fix_sequence(jobname= jobname,
               sequence=seq,
               get_LM_contacts = args.get_LM_contacts,
               copies=1,
               chain_linker=args.chain_linker,
               samples=args.samples,
               masking_rate=args.masking_rate,
               stochastic_mode=args.stochastic_mode,
               outdir=outdir
               )
  
# (jobname = "test",
# sequence = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE",
# num_recycles = 6, #@param ["0", "1", "2", "3", "6", "12"] {type:"raw"}
# get_LM_contacts = False, #@param {type:"boolean"}):  #@param {type:"string"}
# #for homo_oligomer
# copies = 1, #@param {type:"integer"} 
# chain_linker = 25, #@param {type:"number"}
# samples = 8, #@param ["None", "1", "4", "8", "16", "32", "64"] {type:"raw"}
# masking_rate = 0.15, #@param {type:"number"}
# stochastic_mode = "LM", #@param ["LM", "LM_SM", "SM"]
# ):