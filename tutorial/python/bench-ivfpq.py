# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import struct
import sys

if len(sys.argv) != 6:
        print("Usage:\n bench-ivfpq.py base_data.bin query_data.bin M nprobe results.bin")
        quit()

datafile=open(sys.argv[1], "rb")
queryfile=open(sys.argv[2], "rb")
npts,  = struct.unpack('i', datafile.read(4))
ndim,  = struct.unpack('i', datafile.read(4))
nq,    = struct.unpack('i', queryfile.read(4))
ndimq, = struct.unpack('i', queryfile.read(4))

if ndimq != ndim:
        print("Error: Dimensions unequal")
        quit()

print(npts, ndim, nq)

data = np.fromfile(datafile, dtype = np.dtype('f'), count = ndim*npts)
queries = np.fromfile(queryfile, dtype = np.dtype('f'), count = ndim*nq)

data.size
data = np.reshape(data, (npts, ndim))
queries = np.reshape(queries, (nq, ndim))

import faiss                   # make faiss available

ncentroids = 5000
m = int(sys.argv[3])
quantizer = faiss.IndexFlatL2(ndim)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, ndim, ncentroids, m, 8)
index.train(data)
print(index.is_trained)
index.add(data)                  # add vectors to the index
print(index.ntotal)

k = 100                            # we want to see 4 nearest neighbors
#D, I = index.search(data[:5], k) # sanity check
#print(I)
#print(D)

index.nprobe=int(sys.argv[4])
print(index.nprobe)
D, I = index.search(queries, k)     # actual search
gtfile=open(sys.argv[5], "wb")
I = np.reshape(I, nq*k)
D = np.reshape(D, nq*k)
gtfile.write(struct.pack('i',nq))
gtfile.write(struct.pack('i',k))
gtfile.write(struct.pack('i'*nq*k,*I))
gtfile.write(struct.pack('f'*nq*k,*D))
