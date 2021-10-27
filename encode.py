import numpy as np
from numpy import linalg as LA
import torch
from config import MAX_FACES
from torch.autograd import Variable


class Encode:
    """
    Create embedding vector for a 3d model

    Input <str>: path of the preprocessed 3d model in npy format
    Output <List>: normalized embedding vector for that 3d model
    """

    def do_extract(self, path, transformer):
        data = self.prepare(path)
        return self.extract_fea(transformer, *data)

    def extract_fea(self, transformer, centers, corners, normals, neighbor_index):
        centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
        corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
        normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
        neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))
        # get vectors
        feat = list(transformer.get_vector(centers, corners, normals, neighbor_index).tolist())
        return list(feat / LA.norm(feat))

    def prepare(self, path):
        data = np.load(path)
        face = data['faces']
        neighbor_index = data['neighbors']

        # fill for n < max_faces with randomly picked faces
        num_point = len(face)
        if num_point < MAX_FACES:
            fill_face = []
            fill_neighbor_index = []
            for i in range(MAX_FACES - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers[np.newaxis, :, :], corners[np.newaxis, :, :], normals[np.newaxis, :, :], neighbor_index[np.newaxis, :, :]
