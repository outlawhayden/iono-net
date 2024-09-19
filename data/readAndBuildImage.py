import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def readAndBuildImage(iseed=None):
    if iseed is None:
        iseed = 2
    
    matFname = 'data/SAR_AF_ML_toyDataset_etc/radarSeries.mat'
    
    plotImage_Signal_Screen(iseed, matFname)
    plotScreenInMultipleWays(iseed, matFname)
    plotTwoNuStructs(iseed, matFname)

def plotTwoNuStructs(iseed, matFname):
    S = sio.loadmat(matFname)
    print(f'Loading data from {matFname}')
    print(S)
    setup = S['dataset']['meta']['setup']
    nuStructWithSpeckle = S['dataset']['records'][0][iseed]['nuStructs']['withSpeckle']
    PSstruct = S['dataset']['records'][0][iseed]['PSstruct']
    nuStructWithoutSpeckle = S['dataset']['records'][0][iseed]['nuStructs']['withoutSpeckle']
    arg = S['dataset']['meta']['Z']
    
    numPS = len(PSstruct['locs'])
    
    plt.figure(figsize=(10, 6))
    assert np.array_equal(arg, nuStructWithSpeckle['zarg'])
    assert np.array_equal(arg, nuStructWithoutSpeckle['zarg'])
    
    plt.plot(arg, setup['steps']['nu'] * np.abs(nuStructWithoutSpeckle['complVal']), 'r-', linewidth=6, label='without speckle')
    plt.plot(arg, setup['steps']['nu'] * np.abs(nuStructWithSpeckle['complVal']), 'k-', label='with speckle')
    
    plt.legend(fontsize=24)
    plt.xlim([np.min(arg), np.max(arg)])
    plt.grid(True)
    plt.title(f'Test plot: reflectivity (Abs) * step, with and without speckle; point scatterers: {numPS}')
    
    jpgFname = f'reflectivity_record{iseed}.jpg'
    plt.savefig(jpgFname, format='jpeg')

def plotScreenInMultipleWays(iseed, matFname):
    S = sio.loadmat(matFname)
    print(f'Loading data from {matFname}')
    
    setup = S['dataset']['meta']['setup']
    compl_ampls = S['dataset']['records'][0][iseed]['psiParams']['compl_ampls']
    storedPsi = S['dataset']['records'][0][iseed]['storedPsi']
    storedPsi_dd_Val = S['dataset']['records'][0][iseed]['storedPsi_dd_Val']
    
    psiParams = setup['createPsiImplFun'](compl_ampls, setup)
    kPsi = S['dataset']['meta']['kPsi']
    
    zarg = BareboneUtilsIonoAF.create_zarg(setup)
    
    assert np.array_equal(S['dataset']['meta']['S'], storedPsi['arg'])
    assert np.array_equal(S['dataset']['meta']['S'][0::2], zarg)  # only for xi == 0.5

    plt.figure(figsize=(10, 8))

    # Plot psiValDirect
    psiValDirect = buildPsiVals(kPsi, compl_ampls, zarg)
    psiValDirect_dd = np.diff(np.diff(psiValDirect)) / (setup['steps']['nu'])**2
    psiValInternal = psiParams['psiValFun'](zarg, setup)
    
    plt.subplot(2, 1, 1)
    plt.plot(zarg, psiValDirect, 'm-', linewidth=10, label='buildPsiVal')
    plt.plot(storedPsi['arg'], storedPsi['val'], 'k-', linewidth=5, label='storedPsi')
    plt.plot(zarg, psiValInternal, 'g-', linewidth=2, label='psiValFun')

    plt.ylabel('\Psi(s)')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=24)

    plt.subplot(2, 1, 2)
    zarg_dd = zarg[1:-1]
    lightBlue = [0.3010, 0.7450, 0.9330]
    plt.plot(zarg_dd, psiValDirect_dd, '-', linewidth=8, label='diff(diff(psiValDirect))', color=lightBlue)
    plt.plot(storedPsi['arg'], storedPsi_dd_Val, 'k--', linewidth=2, label='storedPsi_dd_Val')
    
    plt.ylabel('\Psi\'\'(s)')
    plt.xlabel('s')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=24)
    
    jpgFname = f'phaseScreen_record{iseed}.jpg'
    plt.savefig(jpgFname, format='jpeg')

def plotImage_Signal_Screen(iseed, matFname):
    plotNos = {'img': 2, 'screen': 3, 'total': 5}
    doShowPeakHeight = True
    
    S = sio.loadmat(matFname, struct_as_record=False, squeeze_me=True)
    print(f'Loading data from {matFname}')
    
    # Access nested structures properly
    dataset = S['dataset']
    meta = dataset.meta
    setup = meta.setup
    
    # Access the required structures
    record = dataset.records[iseed]
    uscStruct = record.uscStruct
    nuStructWithSpeckle = record.nuStructs.withSpeckle
    storedPsi = record.storedPsi
    storedPsi_dd_Val = record.storedPsi_dd_Val
    
    compl_ampls = record.psiParams.compl_ampls
    psiParams = setup.createPsiImplFun(compl_ampls, setup)
    PSstruct = record.PSstruct
    kPsi = meta.kPsi
    
    zarg = BareboneUtilsIonoAF.create_zarg(setup)
    numPS = len(PSstruct.locs)

    plt.figure(figsize=(10, 16))

    plt.subplot(plotNos['total'], 1, 1)
    plt.plot(meta.Z, setup.steps.nu * np.abs(nuStructWithSpeckle.complVal), 'k-', label='Abs')
    plt.legend(fontsize=24)
    plt.xlim([np.min(zarg), np.max(zarg)])
    plt.grid(True)
    plt.title(f'Reflectivity (Abs) * step; point scatterers: {numPS}')

    displayTrueAndZeroImages(plotNos, uscStruct, kPsi, psiParams, zarg, storedPsi, storedPsi_dd_Val, setup)

    jpgFname = f'combinedPanels_record{iseed}.jpg'
    plt.savefig(jpgFname, format='jpeg')

# Additional helper functions will also need to be implemented like buildPsiVals, indicatePointScatterers, etc.

# Usage example
readAndBuildImage(iseed=2)
