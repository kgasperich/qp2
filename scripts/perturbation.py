#!/usr/bin/env python3

import os
from qp_path import QP_SRC

Pert_dir = os.path.join(QP_SRC,"perturbation")

perturbations = []

for filename in [x for x in os.listdir(Pert_dir) if x.endswith(".irp.f")]:

  filename = os.path.join(Pert_dir,filename)
  file = open(filename,'r')
  lines = file.readlines()
  file.close()
  for line in lines:
      buffer = line.lower().lstrip().split()
      if len(buffer) > 1:
        if buffer[0] == "subroutine" and buffer[1].startswith("pt2_"):
          p = (buffer[1].split('(')[0])[4:]
          perturbations.append( p )


if __name__ == '__main__':
  print('Perturbations:')
  for k in perturbations:
    print('* ', k)
