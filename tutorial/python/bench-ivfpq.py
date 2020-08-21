# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import struct
import sys
import faiss                   # make faiss available
import heapq
import multiprocessing

if len(sys.argv) != 8:
        print("Usage:\n bench-ivfpq.py base_data.bin query_data.bin k M nprobe rerank<0/1> results.bin")
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

print('#pts: ', npts, '#queries: ', nq, 'dims: ', ndim)

data = np.fromfile(datafile, dtype = np.dtype('f'), count = ndim*npts)
queries = np.fromfile(queryfile, dtype = np.dtype('f'), count = ndim*nq)

data.size
data = np.reshape(data, (npts, ndim))
queries = np.reshape(queries, (nq, ndim))


ncentroids = 5000
m = int(sys.argv[4])
quantizer = faiss.IndexFlatL2(ndim)
index = faiss.IndexIVFPQ(quantizer, ndim, ncentroids, m, 8)
index.train(data)
print('Index trained? ', index.is_trained)
index.add(data)

k = int(sys.argv[3])
index.nprobe=int(sys.argv[5])
rerank = int(sys.argv[6])

def rerank_with_full_vector(q):
        heap = []
        for i in range(0,rerank_size):
                heapq.heappush(heap, (np.linalg.norm(data[Itmp[q,i]] - queries[q]), Itmp[q,i]))
        for j in range(0,k):
                D[q,j], I[q,j] = heapq.heappop(heap)


if rerank == 1:
        rerank_size = 5*k
        Dtmp, Itmp = index.search(queries, rerank_size) 
        D = np.zeros((nq,k), dtype=float)
        I = np.zeros((nq,k), dtype=int)
        #pool = multiprocessing.Pool(processes=64)
        #pool.map(rerank_with_full_vector, range(0,nq))
        for q in range(0,nq):
                 rerank_with_full_vector(q)
else:
        if rerank != 0:
                print("ERROR: rerank must be 0 or 1. Defaulting to rerank=0")
        D, I = index.search(queries, k)    



gtfile=open(sys.argv[7], "wb")
I = np.reshape(I, nq*k)
D = np.reshape(D, nq*k)
gtfile.write(struct.pack('i',nq))
gtfile.write(struct.pack('i',k))
gtfile.write(struct.pack('i'*nq*k,*I))
gtfile.write(struct.pack('f'*nq*k,*D))
