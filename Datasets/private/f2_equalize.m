function pucchEq = f2_equalize(pucchRx, pucchHest)

pucchEq = pucchRx.*conj(pucchHest)./abs(pucchHest);
dbg = 1;
end

