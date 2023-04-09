using GaussianBasis
using Molecules
using TensorOperations

function RCCSDTgrad(x...)
    RCCSDTgrad(Molecule(), x...)
end

function RCCSDTgrad(mol::Molecule, x...)
    dtype = Options.get("deriv_type")
    if dtype == "analytic"
        throw(FermiException("Invalid or unsupported derivative type for RCCSDT: \"$dtype\""))
    elseif dtype == "findif"
        Fermi.gradient_findif(Fermi.CoupledCluster.RCCSDT, mol, x...)
    else
        throw(FermiException("Invalid or unsupported derivative type: \"$dtype\""))
    end
end

### Analytic graidents go here ###