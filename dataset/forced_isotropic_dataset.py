"""
Torch enabled dataset for working with JHTDB
John Hopkins Turbulenece Database
"""

import numpy as nm
import h5py
import torch
import torch.utils.data as tdata
from pathlib import Path

from dataset import Shapes

xmf_template = """<?xml version="1.0" ?>

<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>

<Xdmf Version="2.0">

  <Domain>

    <Grid Name="Velocity" GridType="Collection" CollectionType="Temporal">

      <Grid Name="Structured Grid" GridType="Uniform">
        <Time Value="{time_step}" />
        <Topology TopologyType="3DRectMesh" NumberOfElements="{resolution} {resolution} {resolution}"/>

        <Geometry GeometryType="VXVYVZ">

          <DataItem Name="Xcoor" Dimensions="{resolution}" NumberType="Float" Precision="4" Format="HDF">
            {h5_file_name}:/xcoor
          </DataItem>

          <DataItem Name="Ycoor" Dimensions="{resolution}" NumberType="Float" Precision="4" Format="HDF">
            {h5_file_name}:/ycoor
          </DataItem>

          <DataItem Name="Zcoor" Dimensions="{resolution}" NumberType="Float" Precision="4" Format="HDF">
            {h5_file_name}:/zcoor
          </DataItem>

        </Geometry>


        <Attribute Name="Velocity" AttributeType="Vector" Center="Node">

          <DataItem Dimensions="{resolution} {resolution} {resolution} 3" NumberType="Float" Precision="4" Format="HDF">
            {h5_file_name}:/u
          </DataItem>
        </Attribute>

      </Grid>
    </Grid>

  </Domain>

</Xdmf>"""


def fill_xmf(time_step, h5_file_name, resolution=128):
    return xmf_template.format(time_step=time_step, h5_file_name=h5_file_name,
                               resolution=resolution)


class ForcedIsotropicDataset(tdata.Dataset):
    """
    Simple class to work with data from forced isotropic 1024 turbulence dataset
    downloaded using data_downloader.

    Getting an item returns time step in simulation (which is not the same as
    index of an item) and the actual data of shape 3 x 128 x 128 x 128
    """
    coors = torch.linspace(0, 6.234098, 128)

    def __init__(self, root_dir=Path("dataset/dltest/"), vel_key="u"):
        self.root_dir = Path(root_dir)
        self.vel_key = vel_key
        self.files = list(self.root_dir.glob("*.h5"))


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        file_path = self.files[idx]
        with h5py.File(file_path, mode='r', swmr=True) as f:
            data = torch.from_numpy(nm.moveaxis(f[self.vel_key], -1, 0))

        time = int(file_path.name.split("_")[0])

        return time, data


def load_cutservice_file(file_path, ret_all=False):
    """
    Load hdf5 file from web cutout service
    """
    with h5py.File(file_path, mode='r', swmr=True) as f:
        keys = list(f)
        vel_keys = [(vel_key, int(vel_key.split("_")[-1]))
                    for vel_key in keys if "Velocity" in vel_key]
        vel_keys, times = zip(*vel_keys)
        coor_keys = [coor_key for coor_key in keys if "coor" in coor_key]

        data = torch.from_numpy(
                nm.stack([nm.moveaxis(f[vel_key], -1, 0) for vel_key in vel_keys]))
        coors = torch.from_numpy(
                nm.stack(
                nm.meshgrid(*[f[coor_key] for coor_key in coor_keys])))
        times = torch.from_numpy(nm.array(times))

    if ret_all:
        return data, coors, times

    return data


def save_cutservice_file(file_path: Path,
                         data: torch.Tensor,
                         coors: torch.Tensor,
                         time_step: int):
    data = data.squeeze().movedim(0, -1).cpu().detach().numpy()
    with h5py.File(file_path, "w") as f:
        f.create_dataset("u", data=data)
        f.create_dataset("xcoor", data=coors)
        f.create_dataset("ycoor", data=coors)
        f.create_dataset("zcoor", data=coors)
        f.attrs.modify("time_step", time_step)


def prepare_coors(resolution):
    x_ = torch.linspace(0, 6.234098, resolution)
    coor = torch.empty((3, resolution, resolution, resolution))
    coor[0], coor[1], coor[2] = torch.meshgrid(x_, x_, x_)
    return coor


def save_paraview_snapshot(path: Path, data, time_step):
    xmf_path = path.with_suffix(".xmf")
    h5_path = path.with_suffix(".h5")
    with open(xmf_path, "w") as f:
        f.write(fill_xmf(time_step, h5_path.name))

    coors = torch.linspace(0, 6.234098, data.shape[-1])

    save_cutservice_file(h5_path, data, ForcedIsotropicDataset.coors, time_step)


if __name__ == '__main__':
    data, coors, times = load_cutservice_file("dataset/prep/isotropic1024coarse_test128_16.h5", ret_all=True)
    # load_cutservice_file("isotropic1024coarse_t100.h5")

    save_paraview_snapshot(Path("dataset/prep/isotropic1024coarse_savetest128_16"), data, 16)
    pass