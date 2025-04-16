


function SoftThresholdingFISTA(in,η,λ)
    out=sign.(in).*max.(abs.(in) .- η*λ,0)

return out;
end





