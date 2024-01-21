import random
import math as m
import torch

from ..utils.get import *
from ..utils.loss_func import *


class map_atomwise():
    def __init__(self, xlo=-90, xhi=90, ylo=-90, yhi=90, zlo=-90, zhi=90, cluster_type_list=None, cluster_center_ordinate_list=None, cluster_angle_list=None):
        self.cluster_type_list = cluster_type_list if cluster_type_list is not None else []
        self.cluster_center_ordinate_list = cluster_center_ordinate_list if cluster_center_ordinate_list is not None else []
        self.cluster_angle_list = torch.tensor(cluster_angle_list) if cluster_angle_list is not None else torch.tensor([])

        self.ncluster = len(self.cluster_type_list)
        self.cluster_natom_list = [get_natom(i) for i in self.cluster_type_list]
        self.cluster_coordinate_list = [get_ordinate_matrix(i) for i in self.cluster_type_list]
        self.cluster_max_dis_list = [get_max_dis(i) for i in self.cluster_coordinate_list]

        self.xlo = xlo
        self.xhi = xhi
        self.ylo = ylo
        self.yhi = yhi
        self.zlo = zlo
        self.zhi = zhi

        # Initialize the mask as an attribute
        self.pair_mask = None


    @staticmethod
    def calculate_coordinate_matrix(cluster_center_list, cluster_coordinate_list, angle_list, device=None):
        rotated_and_translated_coordinates = []

        for i, cluster_coordinates in enumerate(cluster_coordinate_list):
            angle = angle_list[i]


            alpha, beta, gamma = angle[0], angle[1], angle[2]

            ca, sa = torch.cos(alpha), torch.sin(alpha)
            cb, sb = torch.cos(beta), torch.sin(beta)
            cg, sg = torch.cos(gamma), torch.sin(gamma)


            # Directly construct the rotation matrix using operations that maintain the computational graph
            rotation_matrix = torch.stack([
                torch.stack([cb * cg, -cb * sg, sb]),
                torch.stack([sa * sb * cg + ca * sg, -sa * sb * sg + ca * cg, -sa * cb]),
                torch.stack([-ca * sb * cg + sa * sg, ca * sb * sg + sa * cg, ca * cb])
            ], dim=0)

            rotation_matrix.requires_grad_(True)

            # Convert cluster coordinates to PyTorch tensor and apply rotation
            cluster_coordinates_tensor = torch.tensor(cluster_coordinates, dtype=torch.float32, device=device)
            rotated_coordinates = torch.mm(cluster_coordinates_tensor, rotation_matrix)

            # Translate according to the cluster center
            center_coord = cluster_center_list[i].to(device)
            translated_coordinates = rotated_coordinates + center_coord

            rotated_and_translated_coordinates.append(translated_coordinates)

        return rotated_and_translated_coordinates


    def update_pair_mask(self, center_coord_old, center_coord_new, old_cluster_max_dis_list, new_cluster_max_dis_list, device, use_pbc=False):
        m = center_coord_old.shape[0]
        n = center_coord_new.shape[0]
        total_clusters = m + n

        # Concatenate old and new cluster coordinates and ensure they are on the same device
        all_centers = torch.cat([center_coord_old.to(device), center_coord_new.to(device)], dim=0)

        # Calculate pairwise distances considering periodic boundary conditions if required
        if use_pbc:
            # Box dimensions
            box = torch.tensor([self.xhi - self.xlo, self.yhi - self.ylo, self.zhi - self.zlo], device=device)

            # Calculate pairwise differences with periodic boundary conditions
            deltas = all_centers[:, None, :] - all_centers[None, :, :]
            deltas_adjusted = deltas - box * torch.round(deltas / box)

            # Calculate the distances with periodic boundary conditions
            dist_matrix = torch.norm(deltas_adjusted, dim=2)
        else:
            # Calculate pairwise distances in a standard manner
            dist_matrix = torch.norm(all_centers[:, None] - all_centers, dim=2)

        # Create max distance matrix and move it to the correct device
        max_dis = torch.tensor(old_cluster_max_dis_list + new_cluster_max_dis_list, device=device)
        max_dis_matrix = max_dis[:, None] + max_dis

        # Initialize mask on the correct device
        mask = torch.ones((total_clusters, total_clusters), dtype=torch.bool, device=device)

        # Apply criteria
        mask.fill_diagonal_(0)                          # Rule out same cluster pairs
        mask[:m, :m] = 0                               # Exclude pairs between old clusters
        mask = torch.triu(mask)                        # Exclude lower triangular part
        mask[dist_matrix > max_dis_matrix] = 0         # Exclude pairs too far from each other

        self.pair_mask = mask


    def gradient_descend_generate(self, new_cluster_list, rcut, pbc=False, step=100, lr=1, lr_step_size=50, gamma=0.8, gpu=False, track_record=False):
        # Check for GPU availability and set the device
        device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        
        # new_ncluster = len(new_cluster_list)
        # new_cluster_natom_list = [get_natom(i) for i in new_cluster_list]
        new_cluster_coordinate_list = [get_ordinate_matrix(i) for i in new_cluster_list]
        new_cluster_max_dis_list = [get_max_dis(i) for i in new_cluster_coordinate_list]

        center_coord_old = torch.tensor(self.cluster_center_ordinate_list, requires_grad=False, device=device)
        center_coord_new_list = [[random.uniform(self.xlo + r, self.xhi - r), random.uniform(self.ylo + r, self.yhi - r), random.uniform(self.zlo + r, self.zhi - r)] for r in new_cluster_max_dis_list]    
        center_coord_new = torch.tensor(center_coord_new_list, requires_grad=True, device=device)

        # angle_old = torch.tensor(self.cluster_angle_list, requires_grad=False)
        angle_new_list = [[random.uniform(-m.pi, m.pi), random.uniform(-m.pi, m.pi), random.uniform(-m.pi, m.pi)] for i in new_cluster_list]    
        angle_new = torch.tensor(angle_new_list, requires_grad=True, device=device)

        # Create optimizer with different learning rates
        optimizer = torch.optim.SGD([
            {'params': center_coord_new, 'lr': lr},
            {'params': angle_new, 'lr': 0.01 * lr}
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

        old_coords_list = self.calculate_coordinate_matrix(center_coord_old, self.cluster_coordinate_list, self.cluster_angle_list, device)
        track_list = [] if track_record else None

        # with torch.autograd.detect_anomaly():
        for ii in range(step):
            # Update the mask based on the new positions and orientations
            self.update_pair_mask(center_coord_old, center_coord_new, self.cluster_max_dis_list, new_cluster_max_dis_list, device, use_pbc=pbc)

            # Update coordinates for old and new clusters
            new_coords_list = self.calculate_coordinate_matrix(center_coord_new, new_cluster_coordinate_list, angle_new, device)      
            
            new_coords = torch.cat(new_coords_list, dim=0)
            total_loss = 0 if pbc else loss_outbox(new_coords, self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi, device)

            coord_list = old_coords_list + new_coords_list
            if track_record:
                track_list.append(torch.cat(coord_list, dim=0).tolist())

            pairs_to_consider = torch.nonzero(self.pair_mask, as_tuple=True)
            for i, j in zip(pairs_to_consider[0], pairs_to_consider[1]):
                loss_ij = loss_function_atomwise(coord_list[i], coord_list[j], rcut, use_pbc=pbc, xlo=self.xlo, xhi=self.xhi, ylo=self.ylo, yhi=self.yhi, zlo=self.zlo, zhi=self.zhi)
                total_loss += loss_ij

            current_lr = scheduler.get_last_lr()[0]
            print(ii, 'loss=', total_loss.item(), 'lr=', current_lr)

            if total_loss.item() == 0:
                break

            # Gradient descent steps
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()


        self.cluster_type_list = self.cluster_type_list + new_cluster_list
        self.cluster_center_ordinate_list = self.cluster_center_ordinate_list + center_coord_new.tolist()
        self.cluster_angle_list = torch.cat((self.cluster_angle_list, angle_new.to('cpu')), dim=0)
        self.ncluster = len(self.cluster_type_list)
        self.cluster_natom_list = [get_natom(i) for i in self.cluster_type_list]
        self.cluster_coordinate_list = self.cluster_coordinate_list + new_cluster_coordinate_list
        self.cluster_max_dis_list = self.cluster_max_dis_list + new_cluster_max_dis_list

        return track_list if track_record else 0


    def bound_shrink(self, new_cluster_list, rcut, pbc=False, shrink_step=5, shrink_step_size=1, step=100, lr=1, lr_step_size=50, gamma=0.8, gpu=False, track_record=False):
        # Check for GPU availability and set the device
        device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        
        # new_ncluster = len(new_cluster_list)
        # new_cluster_natom_list = [get_natom(i) for i in new_cluster_list]
        new_cluster_coordinate_list = [get_ordinate_matrix(i) for i in new_cluster_list]
        new_cluster_max_dis_list = [get_max_dis(i) for i in new_cluster_coordinate_list]

        center_coord_old = torch.tensor(self.cluster_center_ordinate_list, requires_grad=False, device=device)
        center_coord_new_list = [[random.uniform(self.xlo + r, self.xhi - r), random.uniform(self.ylo + r, self.yhi - r), random.uniform(self.zlo + r, self.zhi - r)] for r in new_cluster_max_dis_list]    
        center_coord_new = torch.tensor(center_coord_new_list, requires_grad=True, device=device)

        # angle_old = torch.tensor(self.cluster_angle_list, requires_grad=False)
        angle_new_list = [[random.uniform(-m.pi, m.pi), random.uniform(-m.pi, m.pi), random.uniform(-m.pi, m.pi)] for i in new_cluster_list]    
        angle_new = torch.tensor(angle_new_list, requires_grad=True, device=device)

        # Create optimizer with different learning rates
        optimizer = torch.optim.SGD([
            {'params': center_coord_new, 'lr': lr},
            {'params': angle_new, 'lr': 0.01 * lr}
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

        old_coords_list = self.calculate_coordinate_matrix(center_coord_old, self.cluster_coordinate_list, self.cluster_angle_list, device)
        track_list, box_list = [], [] if track_record else None

        total_step = 0
        for ii in range(shrink_step):
            print('shrink step is:', ii+1)

            scheduler.step(0)

            self.xlo += shrink_step_size
            self.xhi -= shrink_step_size
            self.ylo += shrink_step_size
            self.yhi -= shrink_step_size
            self.zlo += shrink_step_size
            self.zhi -= shrink_step_size
            print('xlo, xhi, ylo, yhi, zlo, zhi:', self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi)
           
            # with torch.autograd.detect_anomaly():
            for jj in range(step):
                # Update the mask based on the new positions and orientations
                self.update_pair_mask(center_coord_old, center_coord_new, self.cluster_max_dis_list, new_cluster_max_dis_list, device, use_pbc=pbc)

                # Update coordinates for old and new clusters
                new_coords_list = self.calculate_coordinate_matrix(center_coord_new, new_cluster_coordinate_list, angle_new, device)      
                
                new_coords = torch.cat(new_coords_list, dim=0)
                total_loss = 0 if pbc else loss_outbox(new_coords, self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi, device)

                coord_list = old_coords_list + new_coords_list
                if track_record:
                    track_list.append(torch.cat(coord_list, dim=0).tolist())
                    box_list.append([self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi])

                pairs_to_consider = torch.nonzero(self.pair_mask, as_tuple=True)
                for i, j in zip(pairs_to_consider[0], pairs_to_consider[1]):
                    loss_ij = loss_function_atomwise(coord_list[i], coord_list[j], rcut, use_pbc=pbc, xlo=self.xlo, xhi=self.xhi, ylo=self.ylo, yhi=self.yhi, zlo=self.zlo, zhi=self.zhi)
                    total_loss += loss_ij

                total_step += 1
                current_lr = scheduler.get_last_lr()[0]
                print(total_step, jj, 'loss=', total_loss.item(), 'lr=', current_lr)

                if total_loss.item() == 0:
                    break

                # Gradient descent steps
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                print('shrink failed, the finished shrink step is ', ii)
                print('Note! output configuration is failed.')
                break


        self.cluster_type_list = self.cluster_type_list + new_cluster_list
        self.cluster_center_ordinate_list = self.cluster_center_ordinate_list + center_coord_new.tolist()
        self.cluster_angle_list = torch.cat((self.cluster_angle_list, angle_new.to('cpu')), dim=0)
        self.ncluster = len(self.cluster_type_list)
        self.cluster_natom_list = [get_natom(i) for i in self.cluster_type_list]
        self.cluster_coordinate_list = self.cluster_coordinate_list + new_cluster_coordinate_list
        self.cluster_max_dis_list = self.cluster_max_dis_list + new_cluster_max_dis_list

        return [track_list, box_list] if track_record else 0


    def output_xyz(self, path, path_notes=None, coords=None):
        coords_list = coords if coords is not None else torch.cat(self.calculate_coordinate_matrix(torch.tensor(self.cluster_center_ordinate_list), self.cluster_coordinate_list, self.cluster_angle_list), dim=0).tolist()

        file2 = open(path, 'w')
        file2.write('# Cluster_Map Program by NingWang\n')
        file2.write(' ' + str(len(coords_list)) + ' atoms\n')
        file2.write(' 1 atom types\n')
        file2.write(' ' + str(self.xlo) + ' ' + str(self.xhi) + ' xlo xhi\n')
        file2.write(' ' + str(self.ylo) + ' ' + str(self.yhi) + ' ylo yhi\n')
        file2.write(' ' + str(self.zlo) + ' ' + str(self.zhi) + ' zlo zhi\n')
        file2.write('\n')
        file2.write(' Atoms  # atomic\n')
        file2.write('\n')
        for i in range(len(coords_list)):
            file2.write(str(i+1)+' 1 '+' '.join(str(j) for j in coords_list[i])+'\n')
        file2.close()  

        if path_notes is not None:
            file2_notes = open(path_notes, 'w')
            file2_notes.write('# cluster_type_list\n')
            file2_notes.write('# ' + ' '.join(str(j) for j in self.cluster_type_list) + '\n')
            file2_notes.write('# center_ordinate_list\n')
            for i in self.cluster_center_ordinate_list:
                file2_notes.write('# ' + ' '.join(str(j) for j in i) + '\n')    
            file2_notes.write('# angle_list\n')
            for i in self.cluster_angle_list:
                file2_notes.write('# ' + ' '.join(str(j) for j in i) + '\n') 
            file2_notes.close() 
