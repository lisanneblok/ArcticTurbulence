import gsw
import matplotlib.pyplot as plt


def gws_conversions(dataset):
    dataset["P"] = gsw.conversions.p_from_z(dataset["depth"],
                                            dataset["latitude"])
    print(dataset["P"].shape)
    dataset['pot_temp'] = gsw.conversions.pt0_from_t(dataset.S,
                                                     dataset.T, dataset.P)
    # calculate Conservative Temperature from potential temperature
    dataset["CT"] = gsw.CT_from_pt(dataset['S'], dataset["pot_temp"])
    # calculate kappa using gsw_kappa function
    dataset["kappa"] = gsw.kappa(dataset["S"],
                                 dataset["CT"], dataset['S'])
    # calculate N^2 which is calculated for pressure midpoints
    N2, p_mid = gsw.Nsquared(dataset["S"].expand_dims(dim="new_axis", axis=-1), dataset["CT"], dataset["P"],
                             dataset["latitude"])
    #dataset['N2'] = (('DEPTH_MID',), N2)
    #dataset['DEPTH_MID'] = p_mid
    return dataset


def TS_derivative(dataset):
    dataset["dTdz"] = dataset.T.differentiate('depth')
    dataset['dSdz'] = dataset.S.differentiate('depth')
    return dataset


def N2_calc(dataset):
    dataset['N2'] = -9.8/1027*dataset.gamma.differentiate('depth')
    return dataset
#def calculate_hab()
