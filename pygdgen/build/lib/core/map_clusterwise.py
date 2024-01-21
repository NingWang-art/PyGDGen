import random
import numpy as np
import math as m
import torch
import itertools

from ..utils.get import *
from ..utils.loss_func import *


class map_clusterwise():
    def __init__(self, xlo=-90, xhi=90, ylo=-90, yhi=90, zlo=-90, zhi=90, cluster_type_list=None, cluster_center_ordinate_list=None, cluster_angle_list=None):
        self.cluster_type_list = cluster_type_list if cluster_type_list is not None else []
        self.cluster_center_ordinate_list = cluster_center_ordinate_list if cluster_center_ordinate_list is not None else []
        self.cluster_angle_list = cluster_angle_list if cluster_angle_list is not None else []
        self.radius_list = [get_radius(i) for i in self.cluster_type_list]
        self.ncluster = len(self.cluster_type_list)
        self.xlo = xlo
        self.xhi = xhi
        self.ylo = ylo
        self.yhi = yhi
        self.zlo = zlo
        self.zhi = zhi

    def gradient_descend_generate(self, new_cluster_list, step=100, lr=1, lr_step_size=50, gamma=0.8, track_record=False):

        """
        It will output failed configuration if optimition failed eventually,
            while it outputs successful configuration after optimition finished.
        """

        r_old = self.radius_list
        r_new = [get_radius(i) for i in new_cluster_list]
        r = torch.tensor(r_old + r_new, requires_grad=False)
        x_old = torch.tensor(self.cluster_center_ordinate_list, requires_grad=False)
        x_new_list = [[random.uniform(self.xlo + r, self.xhi - r), random.uniform(self.ylo + r, self.yhi - r), random.uniform(self.zlo + r, self.zhi - r)] for r in r_new]    
        x_new = torch.tensor(x_new_list, requires_grad=True)

        optimizer = torch.optim.SGD([x_new], lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)
        # optimizer = torch.optim.Adam([x_new], lr = lr)

        track_list = [] if track_record else None
        for i in range(step):
            if track_record:
                track_list.append(torch.cat((x_old, x_new), dim=0).tolist())
            optimizer.zero_grad()
            loss = loss_function(x_new, x_old, r)

            current_lr = scheduler.get_last_lr()[0]
            print(i, 'loss = ', loss.item(), 'lr=', current_lr)

            if loss.item() == 0:
                break
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for j in range(x_new.shape[0]):
                    x_new[j, 0] = torch.clamp(x_new[j, 0], self.xlo + r_new[j], self.xhi - r_new[j])
                    x_new[j, 1] = torch.clamp(x_new[j, 1], self.ylo + r_new[j], self.yhi - r_new[j])
                    x_new[j, 2] = torch.clamp(x_new[j, 2], self.zlo + r_new[j], self.zhi - r_new[j])                

        x = torch.cat((x_old, x_new), dim=0)
        self.cluster_center_ordinate_list = x.tolist()
        self.cluster_type_list = self.cluster_type_list + new_cluster_list
        self.radius_list = r_old + r_new

        return track_list if track_record else 0

    def bound_shrink(self, shrink_step=5, shrink_step_size=1, learning_step=100, lr=0.3, lr_step_size=50, gamma=0.9, track_record=False):

        """
        Just like gradient_descend_generate(), this function will output failed configuration when optimition failed, while output successful configuration if optimition completed.
        So it's better to determine the optimal limit in the first run, and generate eventual configuration in the second time.
        """
                
        r = torch.tensor(self.radius_list, requires_grad=False)
        x = torch.tensor(self.cluster_center_ordinate_list, requires_grad=True)

        # optimizer = torch.optim.SGD([x], lr = lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
        optimizer = torch.optim.Adam([x], lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

        track_list = [] if track_record else None
        total_step = 0
        for i in range(shrink_step):
            print('shrink step is:', i+1)

            scheduler.step(0)

            self.xlo += shrink_step_size
            self.xhi -= shrink_step_size
            self.ylo += shrink_step_size
            self.yhi -= shrink_step_size
            self.zlo += shrink_step_size
            self.zhi -= shrink_step_size
            print('xlo, xhi, ylo, yhi, zlo, zhi:', self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi)

            for j in range(learning_step):
                with torch.no_grad():
                    for k in range(x.shape[0]):
                        x[k, 0] = torch.clamp(x[k, 0], self.xlo + r[k], self.xhi - r[k])
                        x[k, 1] = torch.clamp(x[k, 1], self.ylo + r[k], self.yhi - r[k])
                        x[k, 2] = torch.clamp(x[k, 2], self.zlo + r[k], self.zhi - r[k])       

                if track_record:
                    track_list.append(x.tolist())

                optimizer.zero_grad()
                loss = loss_function(x, torch.tensor([]), r)

                total_step += 1
                current_lr = scheduler.get_last_lr()[0]
                print(total_step, j, 'loss=', loss.item(), 'lr=', current_lr)

                if loss.item() == 0:
                    break

                loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                print('shrink failure, the finished shrink step is ', i)
                print('Note! output configuration is failed.')
                break

        self.cluster_center_ordinate_list = x.tolist()
        self.cluster_type_list = self.cluster_type_list
        self.radius_list = self.radius_list

        return track_list if track_record else 0
    
    def cylinder_gen_shrink(self, new_cluster_list, cylinder_radius=80, cylinder_span=[-40, 40], shrink_step=5, shrink_step_size=1, learning_step=100, lr=0.3, lr_step_size=50, gamma=0.9, track_record=False):
        
        """
        Just like gradient_descend_generate(), this function will output failed configuration when optimition failed, while output successful configuration if optimition completed.
        So it's better to determine the optimal limit in the first run, and generate eventual configuration in the second time.
        """
        
        r_old = self.radius_list
        r_new = [get_radius(i) for i in new_cluster_list]
        r = torch.tensor(r_old + r_new, requires_grad=False)
        x_old = torch.tensor(self.cluster_center_ordinate_list, requires_grad=False, dtype=float)
        x_new_list = [[random.uniform(cylinder_span[0] + r, cylinder_span[1] - r), random.uniform(-cylinder_radius + r, cylinder_radius - r), random.uniform(-cylinder_radius + r, cylinder_radius - r)] for r in r_new]    
        x_new = torch.tensor(x_new_list, requires_grad=True)

        # optimizer = torch.optim.SGD([x], lr = lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
        optimizer = torch.optim.Adam([x_new], lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

        track_list = [] if track_record else None
        total_step = 0
        for i in range(shrink_step):
            print('shrink step is:', i+1)

            scheduler.step(0)

            cylinder_radius -= shrink_step_size

            for j in range(learning_step):
                with torch.no_grad():
                    for k in range(x_new.shape[0]):
                        x_new[k, 0] = torch.clamp(x_new[k, 0], cylinder_span[0] + r_new[k], cylinder_span[1] - r_new[k])
                        
                        ratio = min(1, (cylinder_radius - r_new[k])/np.sqrt(np.sum(np.square(np.array([x_new[k, 1], x_new[k, 2]])))))
                        x_new[k, 1] *= ratio
                        x_new[k, 2] *= ratio    

                if track_record:
                    track_list.append(torch.cat((x_old, x_new), dim=0).tolist())

                optimizer.zero_grad()
                loss = loss_function(x_new, x_old, r, ignore_xold_loss=True)

                total_step += 1
                current_lr = scheduler.get_last_lr()[0]
                print(total_step, j, 'loss=', loss.item(), 'lr=', current_lr)

                if loss.item() == 0:
                    break

                loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                print('shrink failure, the finished shrink step is ', i)
                print('Note! output configuration is failed.')
                break

        self.cluster_center_ordinate_list = torch.cat((x_old, x_new), dim=0).tolist()
        self.cluster_type_list = self.cluster_type_list + new_cluster_list
        self.radius_list = r_old + r_new

        return track_list if track_record else 0


    def gradient_descend_generate_PBC(self, new_cluster_list, step=100, lr=1, lr_step_size=50, gamma=0.8, track_record=False):
        # Periodic Boundary Condition
        """
        It will output failed configuration if optimition failed eventually,
            while it outputs successful configuration after optimition finished.
        """

        r_old = self.radius_list
        r_new = [get_radius(i) for i in new_cluster_list]
        r = torch.tensor(r_old + r_new, requires_grad=False)
        x_old = torch.tensor(self.cluster_center_ordinate_list, requires_grad=False)
        x_new_list = [[random.uniform(self.xlo + r, self.xhi - r), random.uniform(self.ylo + r, self.yhi - r), random.uniform(self.zlo + r, self.zhi - r)] for r in r_new]    
        x_new = torch.tensor(x_new_list, requires_grad=True)

        optimizer = torch.optim.SGD([x_new], lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)
        # optimizer = torch.optim.Adam([x_new], lr = lr)

        track_list = [] if track_record else None
        for i in range(step):
            if track_record:
                track_list.append(torch.cat((x_old, x_new), dim=0).tolist())

            optimizer.zero_grad()

            lx, ly, lz = self.xhi-self.xlo, self.yhi-self.ylo, self.zhi-self.zlo

            loss = loss_function_PBC(x_new, x_old, r, lx, ly, lz)

            current_lr = scheduler.get_last_lr()[0]
            print(i, 'loss = ', loss.item(), 'lr=', current_lr)

            if loss.item() == 0:
                break
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for j in range(x_new.shape[0]):
                    x_new[j, 0] = (x_new[j, 0] - self.xlo) % lx + self.xlo
                    x_new[j, 1] = (x_new[j, 1] - self.ylo) % ly + self.ylo
                    x_new[j, 2] = (x_new[j, 2] - self.zlo) % lz + self.zlo               

        x = torch.cat((x_old, x_new), dim=0)
        self.cluster_center_ordinate_list = x.tolist()
        self.cluster_type_list = self.cluster_type_list + new_cluster_list
        self.radius_list = r_old + r_new

        return track_list if track_record else 0

    def gradient_descend_generate_PBC_gpu(self, new_cluster_list, step=100, lr=1, lr_step_size=50, gamma=0.8, track_record=False):
        # Move to GPU
        device = 'cuda'

        r_old = self.radius_list
        r_new = [get_radius(i) for i in new_cluster_list]
        r = torch.tensor(r_old + r_new, requires_grad=False).to(device)
        x_old = torch.tensor(self.cluster_center_ordinate_list, requires_grad=False).to(device)
        x_new_list = [[random.uniform(self.xlo + r, self.xhi - r), random.uniform(self.ylo + r, self.yhi - r), random.uniform(self.zlo + r, self.zhi - r)] for r in r_new]    
        x_new = torch.tensor(x_new_list, requires_grad=True, device=device)  # Moved to GPU and made a leaf node

        optimizer = torch.optim.SGD([x_new], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

        track_list = [] if track_record else None
        for i in range(step):
            if track_record:
                track_list.append(torch.cat((x_old, x_new), dim=0).cpu().tolist())

            optimizer.zero_grad()

            lx, ly, lz = self.xhi-self.xlo, self.yhi-self.ylo, self.zhi-self.zlo
            lx, ly, lz = torch.tensor([lx, ly, lz], device=device)  # Moved to GPU

            loss = loss_function_PBC_gpu(x_new, x_old, r, lx, ly, lz)

            current_lr = scheduler.get_last_lr()[0]
            print(i, 'loss = ', loss.item(), 'lr=', current_lr)

            if loss.item() == 0:
                break
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for j in range(x_new.shape[0]):
                    x_new[j, 0] = (x_new[j, 0] - self.xlo) % lx + self.xlo
                    x_new[j, 1] = (x_new[j, 1] - self.ylo) % ly + self.ylo
                    x_new[j, 2] = (x_new[j, 2] - self.zlo) % lz + self.zlo               

        x = torch.cat((x_old, x_new), dim=0)
        self.cluster_center_ordinate_list = x.cpu().tolist()  # Moved back to CPU
        self.cluster_type_list = self.cluster_type_list + new_cluster_list
        self.radius_list = r_old + r_new

        return track_list if track_record else 0


    def bound_shrink_PBC(self, shrink_step=5, shrink_step_size=1, learning_step=100, lr=0.3, lr_step_size=50, gamma=0.9, track_record=False):
        # Periodic Boundary Condition
        """
        Just like gradient_descend_generate(), this function will output failed configuration when optimition failed, while output successful configuration if optimition completed.
        So it's better to determine the optimal limit in the first run, and generate eventual configuration in the second time.
        """
                
        r = torch.tensor(self.radius_list, requires_grad=False)
        x = torch.tensor(self.cluster_center_ordinate_list, requires_grad=True)

        # optimizer = torch.optim.SGD([x], lr = lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
        optimizer = torch.optim.Adam([x], lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

        track_list = [] if track_record else None
        total_step = 0
        for i in range(shrink_step):
            print('shrink step is:', i+1)

            scheduler.step(0)

            self.xlo += shrink_step_size
            self.xhi -= shrink_step_size
            self.ylo += shrink_step_size
            self.yhi -= shrink_step_size
            self.zlo += shrink_step_size
            self.zhi -= shrink_step_size
            print('xlo, xhi, ylo, yhi, zlo, zhi:', self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi)
            lx, ly, lz = self.xhi-self.xlo, self.yhi-self.ylo, self.zhi-self.zlo

            for j in range(learning_step):
                with torch.no_grad():
                    for k in range(x.shape[0]):
                        x[k, 0] = (x[k, 0] - self.xlo) % lx + self.xlo
                        x[k, 1] = (x[k, 1] - self.ylo) % ly + self.ylo
                        x[k, 2] = (x[k, 2] - self.zlo) % lz + self.zlo        

                if track_record:
                    track_list.append(x.tolist())

                optimizer.zero_grad()
                loss = loss_function_PBC(x, torch.tensor([]), r, lx, ly, lz)

                total_step += 1
                current_lr = scheduler.get_last_lr()[0]
                print(total_step, j, 'loss=', loss.item(), 'lr=', current_lr)

                if loss.item() == 0:
                    break

                loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                print('shrink failure, the finished shrink step is ', i)
                print('Note! output configuration is failed.')
                break

        self.cluster_center_ordinate_list = x.tolist()
        self.cluster_type_list = self.cluster_type_list
        self.radius_list = self.radius_list

        return track_list if track_record else 0

    def random_generate(self, new_cluster_list):
        i = 0
        while i < len(new_cluster_list):
            if self.random_place(new_cluster_list[i]) == 0:
                i = i-1
                self.delete_last_place()
            else:
                i = i+1

    def random_place(self, type):
        r = get_radius(type)
        for i in range(10):
            x = random.uniform(self.xlo + r, self.xhi - r)
            y = random.uniform(self.ylo + r, self.yhi - r)
            z = random.uniform(self.zlo + r, self.zhi - r)
            c = 1
            for j in range(len(self.cluster_center_ordinate_list)):
                v1 = np.array([x, y, z])
                v2 = np.array(self.cluster_center_ordinate_list[j])
                if distance(v1, v2) < (self.radius_list[j] + r):
                    c = 0
                    break
            if c == 1:    
                self.cluster_type_list.append(type)
                self.cluster_center_ordinate_list.append([x, y, z])
                self.radius_list.append(r)
                return 1
        return 0

    def delete_last_place(self):
        self.cluster_type_list.pop()
        self.cluster_center_ordinate_list.pop()
        self.radius_list.pop()

    def read_data(self, path):
        data = open(path, 'r').readlines()
        self.cluster_type_list = [int(i) for i in data[1].split(' ')[1:]]
        self.radius_list = [get_radius(i) for i in self.cluster_type_list]
        self.ncluster = len(self.cluster_type_list)
        for i in range(self.ncluster):
            self.cluster_center_ordinate_list.append([float(i) for i in data[3+i].split(' ')[1:]])
            self.cluster_angle_list.append([float(i) for i in data[4+self.ncluster+i].split(' ')[1:]])




class cluster_ordinate():
    def __init__(self, natom=0, matrix=None, xlo=-100, xhi=100, ylo=-100, yhi=100, zlo=-100, zhi=100):
        self.natom = natom
        self.matrix = matrix if matrix is not None else []
        self.xlo = xlo
        self.xhi = xhi
        self.ylo = ylo
        self.yhi = yhi
        self.zlo = zlo
        self.zhi = zhi

    def generate(self, name):  
        self.natom = get_natom(name)
        self.matrix = get_ordinate_matrix(name)

    def generate_configuration(self, type_list, vector_list, angle_list):
        self.generate(type_list[0])
        self.rotate(angle_list[0])
        self.translate(vector_list[0])
        for i in range(1, len(type_list)):
            ordinate_tmp = cluster_ordinate()
            ordinate_tmp.generate(type_list[i])
            ordinate_tmp.rotate(angle_list[i])
            ordinate_tmp.translate(vector_list[i])
            self.concat(ordinate_tmp)

    def rotate(self, angle):
        angle = np.array(angle)
        if len(angle.shape) == 1:
            alpha, beta, gamma = angle[0], angle[1], angle[2] 
            alpha_matrix = np.array([[1, 0, 0], \
                                    [0, m.cos(alpha), -m.sin(alpha)], \
                                    [0, m.sin(alpha), m.cos(alpha)]])
            beta_matrix = np.array([[m.cos(beta), 0, -m.sin(beta)], \
                                    [0, 1, 0], \
                                    [m.sin(beta), 0, m.cos(beta)]])
            gamma_matrix = np.array([[m.cos(gamma), -m.sin(gamma), 0], \
                                    [m.sin(gamma), m.cos(gamma), 0], \
                                    [0, 0, 1]])
            self.matrix = np.dot(self.matrix, alpha_matrix.T)
            self.matrix = np.dot(self.matrix, beta_matrix.T)
            self.matrix = np.dot(self.matrix, gamma_matrix.T)   
        elif len(angle.shape) == 2:
            self.matrix = np.dot(self.matrix, angle.T)

    def translate(self, vector):
        self.matrix = self.matrix + np.array(vector)

    def concat(self, ordinate2):
        self.natom = self.natom + ordinate2.natom
        self.matrix = np.concatenate((self.matrix, ordinate2.matrix),axis=0)
    
    def constraint_recta(self, lo, hi):
        arr = self.matrix
        mask = (arr[:, 0] >= lo[0]) & (arr[:, 0] <= hi[0]) & (arr[:, 1] >= lo[1]) & (arr[:, 1] <= hi[1]) & (arr[:, 2] >= lo[2]) & (arr[:, 2] <= hi[2])
        self.matrix = arr[mask]
        self.natom = len(self.matrix)

    def constraint_rectangular(self, center, length, width, height, reverse=False):
        arr = self.matrix
        # Calculate bounds for the prism
        lower_bound = center - np.array([length / 2.0, width / 2.0, height / 2.0])
        upper_bound = center + np.array([length / 2.0, width / 2.0, height / 2.0])

        # Create a mask for atoms inside the prism
        mask = (arr[:, 0] >= lower_bound[0]) & (arr[:, 0] <= upper_bound[0]) & \
               (arr[:, 1] >= lower_bound[1]) & (arr[:, 1] <= upper_bound[1]) & \
               (arr[:, 2] >= lower_bound[2]) & (arr[:, 2] <= upper_bound[2])
        if reverse:
            mask = ~mask
        self.matrix = arr[mask]
        self.natom = len(self.matrix)

    def constraint_cylinder(self, center, r, height, reverse=False):
        arr = self.matrix
        # Calculate the squared distance from the cylinder axis (assuming it's aligned with the z-axis)
        [cx, cy, cz] = center
        distances_squared = (arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2
        # Create a mask for atoms within the cylinder's radius and height
        mask = (distances_squared <= r ** 2) & (arr[:, 2] >= cz) & (arr[:, 2] <= cz + height)
        # If reverse is True, invert the mask
        if reverse:
            mask = ~mask
        # Apply the mask and update the matrix and natom
        self.matrix = arr[mask]
        self.natom = len(self.matrix)

    def constraint_spheroid(self, center, a, c, reverse=False):
        arr = self.matrix
        mask = ((arr[:, 0] - center[0])**2 + (arr[:, 1] - center[1])**2) / a**2 + (arr[:, 2] - center[2])**2 / c**2 <= 1
        # If reverse is True, invert the mask
        if reverse:
            mask = ~mask
        # Apply the mask and update the matrix and natom
        self.matrix = arr[mask]
        self.natom = len(self.matrix)

    def constraint_torus(self, center, R, r, reverse=False):
        arr = self.matrix
        mask = (((arr[:, 0] - center[0])**2 + (arr[:, 1] - center[1])**2 + (arr[:, 2] - center[2])**2 + R**2 - r**2)**2 - 4*R**2*((arr[:, 0] - center[0])**2 + (arr[:, 1] - center[1])**2)) <= 0
        if reverse:
            mask = ~mask
        self.matrix = arr[mask]
        self.natom = len(self.matrix)

    def constraint_capsule(self, center1, center2, radius, reverse=False):
        arr = self.matrix
        # Calculate masks for two hemispheres
        mask_sphere1 = np.sum((arr - center1)**2, axis=1) <= radius**2
        mask_sphere2 = np.sum((arr - center2)**2, axis=1) <= radius**2

        # Calculate mask for cylinder
        direction = np.array(center2) - np.array(center1)
        unit_direction = direction / np.linalg.norm(direction)
        projection_length = np.dot(arr - center1, unit_direction)
        mask_cylinder = (projection_length >= 0) & (projection_length <= np.linalg.norm(direction)) & (np.sum(np.cross(arr - center1, unit_direction)**2, axis=1) <= radius**2)

        mask = mask_sphere1 | mask_sphere2 | mask_cylinder
        if reverse:
            mask = ~mask
        self.matrix = arr[mask]
        self.natom = len(self.matrix)

    def constraint_dumbbell(self, center1, center2, radius_sphere1, radius_sphere2, radius_cylinder, reverse=False):
        arr = self.matrix
        # Calculate masks for two spheres with different radii
        mask_sphere1 = np.sum((arr - center1)**2, axis=1) <= radius_sphere1**2
        mask_sphere2 = np.sum((arr - center2)**2, axis=1) <= radius_sphere2**2

        # Calculate mask for cylinder with radius_cylinder
        direction = np.array(center2) - np.array(center1)
        unit_direction = direction / np.linalg.norm(direction)
        projection_length = np.dot(arr - center1, unit_direction)
        mask_cylinder = (projection_length >= 0) & (projection_length <= np.linalg.norm(direction)) & (np.sum(np.cross(arr - center1, unit_direction)**2, axis=1) <= radius_cylinder**2)

        mask = mask_sphere1 | mask_sphere2 | mask_cylinder
        if reverse:
            mask = ~mask
        self.matrix = arr[mask]
        self.natom = len(self.matrix)


    def constraint_tetrahedron(self, center, side_length, reverse=False):
        arr = self.matrix

        # Side length of the cube in which the tetrahedron is inscribed
        a = side_length / np.sqrt(2) / 2

        # Calculate the vertices of the regular tetrahedron
        v0 = center + a * np.array([-1, -1, -1])
        v1 = center + a * np.array([1, 1, -1])
        v2 = center + a * np.array([-1, 1, 1])
        v3 = center + a * np.array([1, -1, 1])

        # Function to calculate the signed volume of a tetrahedron
        def signed_volume(a, b, c, d):
            return np.dot(np.cross(b - a, c - a), d - a) / 6.0

        # Function to check if a point is inside the tetrahedron
        def is_inside_tetrahedron(point, v0, v1, v2, v3):
            vol0 = signed_volume(point, v1, v2, v3)
            vol1 = signed_volume(v0, point, v2, v3)
            vol2 = signed_volume(v0, v1, point, v3)
            vol3 = signed_volume(v0, v1, v2, point)

            # Check if all volumes have the same sign
            return (vol0 >= 0 and vol1 >= 0 and vol2 >= 0 and vol3 >= 0) or \
                   (vol0 <= 0 and vol1 <= 0 and vol2 <= 0 and vol3 <= 0)

        mask = np.array([is_inside_tetrahedron(atom, v0, v1, v2, v3) for atom in arr])
        if reverse:
            mask = ~mask

        self.matrix = arr[mask]
        self.natom = len(self.matrix)


    def get_PBC_coordinate(self):
        lx, ly, lz = self.xhi - self.xlo, self.yhi - self.ylo, self.zhi - self.zlo
        for v in self.matrix:
            v[0] = (v[0] - self.xlo) % lx + self.xlo
            v[1] = (v[1] - self.ylo) % ly + self.ylo
            v[2] = (v[2] - self.zlo) % lz + self.zlo      

    def generate_special_wulff(self, miller_index, distance):
        # distance > 0
        pos_neg_combinations = [np.array([num, -num]) for num in miller_index]
        arrangements = list(itertools.product(*pos_neg_combinations))
        millers = np.array(list(set([item for i in arrangements for item in list(itertools.permutations(i))])))
        print(millers)
        d = np.sqrt(np.sum(np.square(miller_index))) * distance

        detemines = np.max(np.abs(np.dot(self.matrix, millers.T)), axis=1)
        
        indices = np.where(detemines > d)

        self.matrix = np.delete(self.matrix, indices, axis=0)
        self.natom -= len(indices[0])

        print('remove', len(indices[0]), 'atoms, ', self.natom, 'left.')


    def output_xyz(self, path, cluster_list=None, center_ordinate_list=None, angle_list=None, path_notes=None):
        file2 = open(path, 'w')
        file2.write('# Cluster_Map Program by NingWang\n')
        file2.write(' ' + str(self.natom) + ' atoms\n')
        file2.write(' 1 atom types\n')
        file2.write(' ' + str(self.xlo) + ' ' + str(self.xhi) + ' xlo xhi\n')
        file2.write(' ' + str(self.ylo) + ' ' + str(self.yhi) + ' ylo yhi\n')
        file2.write(' ' + str(self.zlo) + ' ' + str(self.zhi) + ' zlo zhi\n')
        file2.write('\n')
        file2.write(' Atoms  # atomic\n')
        file2.write('\n')
        for i in range(self.natom):
            file2.write(str(i+1)+' 1 '+' '.join(str(j) for j in self.matrix[i, :])+'\n')
        file2.close()  

        if path_notes is not None:
            file2_notes = open(path_notes, 'w')
            file2_notes.write('# cluster_type_list\n')
            file2_notes.write('# ' + ' '.join(str(j) for j in cluster_list) + '\n')
            file2_notes.write('# center_ordinate_list\n')
            for i in center_ordinate_list:
                file2_notes.write('# ' + ' '.join(str(j) for j in i) + '\n')    
            file2_notes.write('# angle_list\n')
            for i in angle_list:
                file2_notes.write('# ' + ' '.join(str(j) for j in i) + '\n') 
            file2_notes.close() 









