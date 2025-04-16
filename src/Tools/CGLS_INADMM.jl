function CGLSInADMM(η0, Ahk, operators, parameters; μ::Union{AbstractFloat,Vector}=0.5, Ni=100, tol=1e-6, history=true)

    if history
        println("")
        println(" =====================================================================================")
        println("                       CGLS for (AAᴴ + μI)η= Ahₖ")
        println(" =====================================================================================")
        println("")
    end

    if μ isa AbstractFloat
        μ = fill(μ, Ni)
    elseif μ isa Vector
        μ = float.(μ)
    else
        error("μ should be either a Float or a Vector.")
    end

    #Initialize history tracking
    J_history = Float64[]
    Jm_history = Float64[]
    Jr_history = Float64[]
    grad_norm_history = Float64[]
    res_norm_history= Float64[]
    Jm0=norm(η0,2)^2;
    Jr0=norm(Ahk,2)^2;
    #J0 = Jm0 + μ[begin]*Jr0

    if history
        header = "k         ||y-Ax||²₂              ||x||²₂                   μ                   J"
        println(""); 
        println(header);
      #  @printf("%3.0f %20.10e %20.10e  %20.10e %20.10e\n",0,Jm0, Jr0,μ[begin],J0);
    end


    η = copy(η0)
    Aᴴη = LinearOperator(η, operators, parameters, adj=true)
    AAᴴη = LinearOperator(Aᴴη, operators, parameters, adj=false)
    y = AAᴴη + μ[begin] .* η
    r = Ahk - y
    ∇J = copy(r)
    d = copy(∇J)

    #Jm0=norm(dobs,2)^2;
    #Jr0=norm(m0,2)^2;
    #J0 = Jm0 + μ[begin]*Jr0


    k = 0

    # Main loop
    while k < Ni

        k += 1

        Aᴴd = LinearOperator(d, operators, parameters, adj=true)
        AAᴴd = LinearOperator(Aᴴd, operators, parameters, adj=false)
        sl_denom = dot(d, AAᴴd) + μ[k] * dot(d, d)

        if abs(sl_denom) < tol
            println("sl_denom ≈ 0. It is not possible to compute the step length. Loop ending at iteration $k")
            break
        end

        α = dot(∇J, ∇J) / sl_denom
        η_new = η + α * d

        Aᴴη = LinearOperator(η_new, operators, parameters, adj=true)
        AAᴴη = LinearOperator(Aᴴη, operators, parameters, adj=false)
        r = Ahk .- (AAᴴη + μ[k] .* η_new)
        ∇J_new = copy(r)

        β_k = dot(∇J_new, ∇J_new) / dot(∇J, ∇J)
        d = ∇J_new + β_k * d


        Jmk= norm(r,2)^2; #New misfit
        Jrk= norm(η_new,2)^2; #New model norm
        Jk = Jmk +μ[k]*Jrk; # New cost function valu

        res_norm = norm(r, 2)
        grad_norm= norm(∇J_new,2);


        push!(Jm_history, Jmk); #save misfit at each iteration
        push!(Jr_history, Jrk); #save regularization at each iteration
        push!(J_history,Jk); #save objective at each iteration
        push!(grad_norm_history, grad_norm) # save grad norm at each iteration
        push!(res_norm_history, res_norm) # save grad norm at each iteration



        if history && k <Ni
            @printf("%3.0f %20.10e %20.10e  %20.10e %20.10e\n", k, Jmk, Jrk, μ[k], Jk)
        end        


        if res_norm < tol && grad_norm < tol

            if history
                println("CGLS converged at iteration $k with residual norm: $res_norm and gradient norm: $grad_norm .")
            end
            break
        end

        if length(J_history) > 1 && J_history[end] > eps()

            ΔJ= abs((J_history[end] - J_history[end-1]) / J_history[end])

            if round(ΔJ,digits=8) < tol
               println("Loop for CGLS stopped at $k iterations.")
               println("REASON: ")
               println(" ΔJ = $ΔJ   is < than the established tolerance = $tol used.")
               break
            end			
        end
        η = η_new;
        ∇J = ∇J_new;
    end




    return η, J_history,k  #grad_norm_history
end
