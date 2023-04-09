"""
    Fermi.CoupledCluster.cc_update_T1!(newT1::AbstractArray{T,2}, T1::AbstractArray{T,2}, T2::AbstractArray{T,4}, moints::IntegralHelper{T,O}, 
                      alg::RCCSDTa) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}

Compute DᵢₐTᵢₐ from old T1 and T2 amplitudes. Final updated T1 amplitudes can be obtained by applying the denominator 1/Dᵢₐ.
NOTE: It does not include off-diagonal Fock contributions: See `od_cc_update_T1`.
"""
function cc_update_T1!(newT1::AbstractArray{T,2}, T1::AbstractArray{T,2}, T2::AbstractArray{T,4}, T3::AbstractArray{T, 6},
    moints::IntegralHelper{T,E,O},alg::RCCSDTa) where {T<:AbstractFloat, E<:AbstractERI, O<:AbstractRestrictedOrbitals}

    ## Start by adding the T1 terms from CCSD
    cc_update_T1!(newT1, T1, T2, moints, RCCSDa())

    ## include the Triples contribution.
    Voovv = moints["OOVV"]
    TWO = T(2)
    @tensoropt (i=>x, j=>x, k=>x, l=>x, a=>10x, b=>10x, c=>10x, d=>10x) begin
        newT1[i,a] -= (TWO * Voovv[j,k,b,c] - Voovv[j,k,c,b])  * (T3[j,k,i,b,a,c] - T3[j,k,i,b,c,a])
    end

end

"""
    Fermi.CoupledCluster.cc_update_T2!(newT2::AbstractArray{T,4}, T1::AbstractArray{T,2}, T2::AbstractArray{T,4}, moints::IntegralHelper{T,O}, 
                    alg::RCCSDTa) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}

Compute Dᵢⱼₐᵦ⋅Tᵢⱼₐᵦ from old T1 and T2 amplitudes. Final updated T2 amplitudes can be obtained by applying the denominator 1/Dᵢⱼₐᵦ.
NOTE: It does not include off-diagonal Fock contributions: See `od_cc_update_T2`.
"""
function cc_update_T2!(newT2::AbstractArray{T,4}, T1::AbstractArray{T,2}, T2::AbstractArray{T,4}, T3::AbstractArray{T, 6},
     moints::IntegralHelper{T,E,O}, alg::RCCSDTa) where {T<:AbstractFloat, E<:AbstractERI, O<:AbstractRestrictedOrbitals}

    cc_update_T2!(newT2, T1, T2, moints, RCCSDa())
    
    @show norm(newT2)
    ## Add triples contribution
    Vooov, Vovvv, Voovv = moints["OOOV"], moints["OVVV"], moints["OOVV"]
    Fia = moints["Fia"]

    TWO = T(2)
    tempT2 = similar(newT2)
    @tensoropt (i=>x, j=>x, k=>x, l=>x, a=>10x, b=>10x, c=>10x, d=>10x) begin

        tempT2[i,j,a,b] = (TWO * Vovvv[k,a,d,c] - Vovvv[k,a,c,d]) * T3[i,j,k,c,b,d]
        tempT2[i,j,a,b] -= Vovvv[k,a,d,c] * T3[i,j,k,c,d,b]
        tempT2[i,j,a,b] -= (TWO * Vooov[k,l,i,c] - Vooov[l,k,i,c]) * T3[k,j,l,a,b,c] 
        tempT2[i,j,a,b] -= - Vooov[k,l,i,c] * T3[k,j,l,a,c,b]
                
        tempT2[i,j,a,b] -= Fia[k,c] * (T3[i,k,j,a,b,c] - T3[i,k,j,a,c,b])

        tempT2[i,j,a,b] = tempT2[i,j,a,b] + tempT2[j,i,b,a]
        newT2[i,j,a,b] += tempT2[i,j,a,b]
    end

    # If using CCSDT1 you should return newT2 here.
    #Array o2_abij;
    v_asym = similar(Voovv)
    @tensoropt (i=>x, j=>x, k=>x, l=>x, a=>10x, b=>10x, c=>10x, d=>10x) begin
        v_asym[k,l,c,d] = (TWO * Voovv[k,l,c,d] - Voovv[l,k,c,d])
        tempT2[i,j,a,b] =  (v_asym[k,l,c,d] * (T3[i,j,k,a,b,c] - T3[i,j,k,a,c,b])) * T1[l,d]
        tempT2[i,j,a,b] -= (v_asym[k,l,c,d] * T3[i,k,l,a,c,b] - Voovv[k,l,c,d] * T3[i,k,l,c,a,b]) * T1[j,d]
        tempT2[i,j,a,b] -= (v_asym[k,l,c,d] * T3[i,j,k,a,d,c] - Voovv[k,l,c,d] * T3[i,j,k,c,d,a]) * T1[l,b]

        tempT2[i,j,a,b] = tempT2[i,j,a,b] + tempT2[j,i,b,a]
        newT2[i,j,a,b] += tempT2[i,j,a,b]
    end
    @show norm(newT2)
end

function cc_update_T3_v4_term!(newT2::AbstractArray{T,4}, T1::AbstractArray{T,2}, T2::AbstractArray{T,4}, T3::AbstractArray{T, 6},
    moints::IntegralHelper{T,Chonky,O}, alg::RCCSDTa) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}
                
    Vvvvv = moints["VVVV"]
    @tensoropt (i=>x, j=>x, k=>x, l=>x, a=>10x, b=>10x, c=>10x, d=>10x) begin
        τ[i,j,a,b] := T2[i,j,a,b] + T1[i,a]*T1[j,b]
        newT2[i,j,a,b] += τ[i,j,c,d]*Vvvvv[c,a,d,b]
    end
end

function cc_update_T3_v4_term!(newT2::AbstractArray{T,4}, T1::AbstractArray{T,2}, T2::AbstractArray{T,4}, T3::AbstractArray{T, 6},
    moints::IntegralHelper{T,E,O}, alg::RCCSDTa) where {T<:AbstractFloat, E<:AbstractDFERI, O<:AbstractRestrictedOrbitals}
                
    Bvv = moints["BVV"]
    @tensor τ[i,j,a,b] := T2[i,j,a,b] + T1[i,a]*T1[j,b]

    o_size = size(T1, 1)
    v_size = size(T1, 2)
    cdb = zeros(T, v_size, v_size, v_size)
    newT2a = zeros(T, o_size, o_size, v_size)
    for a = 1:v_size
        Ba = @views Bvv[:,:,a]
        @tensor cdb[c,d,b] = Ba[Q,c]*Bvv[Q,d,b]
        @tensor newT2a[i,j,b] = τ[i,j,c,d]*cdb[c,d,b]
        newT2[:,:,a,:] += newT2a
    end
end

"""
    Fermi.CoupledCluster.update_amp!(newT1::AbstractArray{T,2}, newT2::Array{T,4}, T1::Array{T, 2}, T2::Array{T, 4}, moints::IntegralHelper{T,O}, 
                     alg::A) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}

Computes new T1 and T2 amplitudes from old ones. It assumes Restricted Hartree-Fock reference.
"""
function update_amp!(newT1::AbstractArray{T,2}, newT2::AbstractArray{T,4}, newT3::AbstractArray{T, 6}, T1::AbstractArray{T, 2}, T2::AbstractArray{T, 4}, 
    T3::AbstractArray{T, 6}, moints::IntegralHelper{T,E,RHFOrbitals}, 
                     alg::RCCSDTa) where {T<:AbstractFloat,E<:AbstractERI}

    # Clean the arrays
    fill!(newT1, 0.0)
    fill!(newT2, 0.0)

    # Get new amplitudes
    cc_update_T1!(newT1, T1, T2, T3, moints, alg)
    cc_update_T2!(newT2, T1, T2, T3, moints, alg)

    # Orbital energies line
    if haskey(moints.cache, "D1")
        d = moints["D1"]
    else
        Fd = moints["Fd"]
        ndocc = moints.molecule.Nα
        frozen = Options.get("drop_occ")
        inac = Options.get("drop_vir")
        ϵo = Fd[(1+frozen:ndocc)]
        ϵv = Fd[(1+ndocc):end-inac]

        d = [ϵo[i]-ϵv[a] for i=eachindex(ϵo), a=eachindex(ϵv)]
        moints["D1"] = d
    end

    if haskey(moints.cache, "D2")
        D = moints["D2"]
    else
        Fd = moints["Fd"]
        ndocc = moints.molecule.Nα
        frozen = Options.get("drop_occ")
        inac = Options.get("drop_vir")
        ϵo = Fd[(1+frozen:ndocc)]
        ϵv = Fd[(1+ndocc):end-inac]

        D = [ϵo[i]+ϵo[j]-ϵv[a]-ϵv[b] for i=eachindex(ϵo), j=eachindex(ϵo), a=eachindex(ϵv), b=eachindex(ϵv)]
        moints["D2"] = D
    end

    # Orbital energies line
    d, D = moints["D1"], moints["D2"]

    newT1 ./= d
    newT2 ./= D
end


## Methods for non-orthogonal CC methods (need to compute off diagonal because not using HF orbitals)
# """
#     Fermi.CoupledCluster.od_cc_update_T1!(newT1::AbstractArray{T,2}, T1::AbstractArray{T,2}, T2::AbstractArray{T,4}, moints::IntegralHelper{T,O}, 
#                       alg::RCCSDTa) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}

# Compute non-diagonal Fock contributions to DᵢₐTᵢₐ from old T1 and T2 amplitudes. Final updated T1 amplitudes 
# can be obtained by applying the denominator 1/Dᵢₐ.
# See also: `cc_update_T1`.
# """
# function od_cc_update_T1!(newT1::AbstractArray{T,2}, T1::AbstractArray{T,2}, T2::AbstractArray{T,4}, T3::AbstractArray{T, 6},
#      moints::IntegralHelper{T,E,O}, alg::RCCSDTa) where {T<:AbstractFloat, E<:AbstractERI, O<:AbstractRestrictedOrbitals}

#     od_cc_update_T1!(newT1, T1, T2, moints, RCCSDa())

#     # Add triples terms
#     throw(ErrorException)
# end
# """
#     Fermi.CoupledCluster.od_cc_update_T2!(newT2::AbstractArray{T,4}, T1::AbstractArray{T,2}, T2::AbstractArray{T,4}, moints::IntegralHelper{T,O}, 
#                     alg::RCCSDTa) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}

# Compute non-diagonal Fock contribution to Dᵢⱼₐᵦ⋅Tᵢⱼₐᵦ from old T1 and T2 amplitudes. Final updated T2 amplitudes 
# can be obtained by applying the denominator 1/Dᵢⱼₐᵦ. See also: `cc_update_T2`.
# """
# function od_cc_update_T2!(newT2::AbstractArray{T,4}, T1::AbstractArray{T,2}, T2::AbstractArray{T,4}, T3::AbstractArray{T, 6},
#      moints::IntegralHelper{T,E,O}, alg::RCCSDTa) where {T<:AbstractFloat, E<:AbstractERI, O<:AbstractRestrictedOrbitals}

#     # # Include non-RHF terms
#     # fov = moints["Fia"]
#     # foo = moints["Fij"]
#     # fvv = moints["Fab"]
#     # ONE = one(T)
#     # @tensoropt (i=>x, j=>x, k=>x, l=>x, a=>10x, b=>10x, c=>10x, d=>10x) begin
#     #     P_OoVv[i,j,a,b] := -ONE*foo[i,k]*T2[k,j,a,b]
#     #     P_OoVv[i,j,a,b] += fvv[c,a]*T2[i,j,c,b]
#     #     P_OoVv[i,j,a,b] -= fov[k,c]*T1[i,c]*T2[k,j,a,b]
#     #     P_OoVv[i,j,a,b] -= fov[k,c]*T1[k,a]*T2[i,j,c,b]
#     #     newT2[i,j,a,b] += P_OoVv[i,j,a,b] + P_OoVv[j,i,b,a]
#     # end
#     # #Include RHF terms
#     # cc_update_T2!(newT2, T1, T2, moints, alg)
#     od_cc_update_T2!(newT2, T1, T2, moints, RCCSDa())

#     ## Get off diagonal triples contributions
# end

# """
#     Fermi.CoupledCluster.update_amp!(newT1::AbstractArray{T,2}, newT2::Array{T,4}, T1::Array{T, 2}, T2::Array{T, 4}, moints::IntegralHelper{T,O}, 
#                      alg::A) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}

# Computes new T1 and T2 amplitudes from old ones. It assumes arbitrary restricted orbitals.
# """
# function update_amp!(newT1::AbstractArray{T,2}, newT2::AbstractArray{T,4},  newT3::AbstractArray{T, 6}, T1::AbstractArray{T, 2}, T2::AbstractArray{T, 4}, T3::AbstractArray{T, 6}, moints::IntegralHelper{T,E,O}, alg::RCCSDTa) where {T<:AbstractFloat, E<:AbstractERI, O<:AbstractRestrictedOrbitals}

#     # Clean the arrays
#     fill!(newT1, 0.0)
#     fill!(newT2, 0.0)

#     # Get new amplitudes
#     od_cc_update_T1!(newT1, T1, T2, T3, moints, alg)
#     od_cc_update_T2!(newT2, T1, T2, T3, moints, alg)

#     # Orbital energies line
#     if haskey(moints.cache, "D1")
#         d = moints["D1"]
#     else
#         Fd = moints["Fd"]
#         ndocc = moints.molecule.Nα
#         frozen = Options.get("drop_occ")
#         inac = Options.get("drop_vir")
#         ϵo = Fd[(1+frozen:ndocc)]
#         ϵv = Fd[(1+ndocc):end-inac]

#         d = [ϵo[i]-ϵv[a] for i=eachindex(ϵo), a=eachindex(ϵv)]
#         moints["D1"] = d
#     end

#     if haskey(moints.cache, "D2")
#         D = moints["D2"]
#     else
#         Fd = moints["Fd"]
#         ndocc = moints.molecule.Nα
#         frozen = Options.get("drop_occ")
#         inac = Options.get("drop_vir")
#         ϵo = Fd[(1+frozen:ndocc)]
#         ϵv = Fd[(1+ndocc):end-inac]

#         D = [ϵo[i]+ϵo[j]-ϵv[a]-ϵv[b] for i=eachindex(ϵo), j=eachindex(ϵo), a=eachindex(ϵv), b=eachindex(ϵv)]
#         moints["D2"] = D
#     end

#     newT1 ./= d
#     newT2 ./= D
# end