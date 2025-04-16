
function ADMM(m0,dobs,operators,parameters; ρ=0.1, μ=0.5,tolin=1e-2, tolout=1e-4, Ni=5,Ne=50, α0=0.5,history=true, ls_solver="cgls")


    if ls_solver == "cgls"

        m, J,Jls,time = ADMM_CGLS(m0,dobs,operators,parameters; ρ=ρ, μ=μ,tolin=tolin, tolout=tolout, Ni=Ni,Ne=Ne, history=history)

    elseif ls_solver == "sd"

        m, J,Jls= ADMM_SD(m0,dobs,operators,parameters; ρ=ρ, μ=μ,tolin=tolin, tolout=tolout, Ni=Ni,Ne=Ne, history=history)

    else
        error("No more implementations for now.")

    end
    
    
    return m,J,Jls,time

end



function ADMM_CGLS(m0,dobs,operators,parameters; ρ=0.1, μ=0.5,tolin=1e-2, tolout=1e-4, Ni=5,Ne=50, history=true)


    if history
        println("")
        println("=====================================================================================================================")
        println("                            Alternating  Direction Method of Multipliers (ADMM)")
        println("=====================================================================================================================")
        println("")

    end




    if ρ isa AbstractFloat
        ρ = fill(ρ, Ne) 
    elseif ρ isa Vector
        ρ = float.(ρ) 
    else
        error("ρ should be either a Float or a Vector.")
    end

    if μ isa AbstractFloat
        μ = fill(μ, Ne)  # Create a vector of size Ni
    elseif μ isa Vector
        μ = float.(μ)  # Ensure it's Float64
    else
        error("μ should be either a Float or a Vector.")
    end
    
    


    # Initialize history tracking
    Jls = Float64[]
    J_history = Float64[]
    Jm_history = Float64[]
    Jr_history = Float64[]
    prim_res_history = Float64[]
    dual_res_history = Float64[]
    m_diff_history = Float64[]
    get_time = Float64[]
    cgls_iteration=Float64[];

    
    #Initialize variables
    u=zeros(eltype(m0),size(m0)); 
    z=zeros(eltype(m0),size(m0)); 
    w=zeros(eltype(m0),size(m0));
    #x=zeros(eltype(m0),size(m0)); 

 

    #Initial models and parameters that change

    #α=α0;
    x = m0;# Initial model
    Jm0=norm(dobs,2)^2;
    Jr0=norm(m0,2)^2;
    J0= Jm0 + μ[begin]Jr0;


   if history
    header = "   k         J              ||y-Ax||²₂             ||x||₁                 ρ                 μ                  ||r||₂                    ||s||₂ "
    println(""); 
    println(""); 
    println(header);
   end


    k=0; #counter

    #Main loop

    
    while k < Ne



        gt= @elapsed begin


        k += 1  # Increment iteration counter
        
        w_old=w;
        b= dobs -1*LinearOperator(z .- u ,operators, parameters, adj=false);
        w, Jls, cgls_iter=CGLS(w_old,b,operators, parameters; μ=ρ[k], Ni=Ni, tol=tolin, history=false)
        println("CGLS iterations performed = $cgls_iter")

        x_old=x;
        x=  w .+ (z .- u);
        z_old = z;  # Store previous z for dual residual
        z= SoftThresholdingADMM( x .+ u, μ[k] , ρ[k]); # z-update
          
        u= u .+ (x .- z);

        end

        prim_res = x - z; #Primal residual
        dual_res = ρ[k]*(z - z_old); # Dual residual;

        
        dpred= LinearOperator(x,operators,parameters,adj=false); #predicted data at iteration k
        Jmk= norm(dobs .- dpred,2)^2;  #||Axₖ - y||₂²
        Jrk= sum(abs.(z));  #||zₖ||₁ at
        Jk = Jmk + μ[k]*Jrk; #objective function at iteration k

        m_diff = norm(x - x_old, 2) # model norm difference 
        res_norm = norm(Jmk, 2) #norma del residuo
        prim_res_norm= norm(prim_res, 2)
        dual_res_norm= norm(dual_res, 2)


           #update rho
        if k< Ne && prim_res_norm > 10*dual_res_norm
            ρ[k+1] *= 2;
            u = u./2; # u = 1/ρ *y
            
        elseif k< Ne && dual_res_norm > 10*prim_res_norm
            ρ[k+1] /= 2;
            u = 2*u;

        end
        




        push!(prim_res_history, prim_res_norm)
        push!(dual_res_history, dual_res_norm)
        push!(Jm_history, Jmk); #save misfit at each iteration
        push!(Jr_history, Jrk); #save regularization at each iteration
        push!(J_history,Jk); #save objective
        push!(m_diff_history, m_diff)
        push!(get_time,gt)  #save time
        push!(cgls_iteration,cgls_iter);



        if history && k <= Ne
            println("---------------------------------------------------------------------------------------------------------------------------------------------------------"); 
            @printf("%3.0f %20.10e %20.10e %20.10e  %20.10e %20.10e %20.10e %20.10e\n",k,Jk,Jmk,Jrk,ρ[k],μ[k],prim_res_norm,dual_res_norm);
        end


        if res_norm < tolin

            if history
                println("CGLS converged at iteration $k with residual norm: $res_norm.")
            end
            break
        end

        if length(J_history) > 1 && J_history[end] < eps();

            ΔJ= abs((J_history[end] - J_history[end-1]) / J_history[end])

            if round(ΔJ,digits=8) < tolout
               println("Outer-loop for ADMM stopped at $k iterations.")
               println("REASON: ")
               println(" ΔJ = $ΔJ   is < than the established outer loop tolerance = $tolout used.")
               break
            end			
        end
        
    

    
    end


       #possible outputs
       
        #misfit_iteration= res_norm; # residual norm or misfit => ensure you fit the data
        #J_history; objective function 
        #Jm_history; # data fidelity term for χ² test.
        #Jr_history; # model norm fidelity term for χ² test.
        #prim_res_history; primal variable residual
        #dual_res_history; dual variable residual
        #m_diff_history; model difference 


    return x, J_history, Jls, get_time #, r_history, s_history, m_diff_history

end


function ADMM_SD(m0,dobs,operators,parameters; ρ=0.1, μ=0.5, α0= 0.1, tolin=1e-2, tolout=1e-4, Ni=5,Ne=50, history=true)


    if history
        println("")
        println("=====================================================================================================================")
        println("                            Alternating  Direction Method of Multipliers (ADMM)")
        println("=====================================================================================================================")
        println("")

    end


    # Initialize history tracking
    Jls = Float64[]
    J_history = Float64[]
    Jm_history = Float64[]
    Jr_history = Float64[]
    prim_res_history = Float64[]
    dual_res_history = Float64[]
    m_diff_history = Float64[]
    
    
    #Initialize variables
    u=zeros(eltype(m0),size(m0)); 
    z=zeros(eltype(m0),size(m0)); 
    w=zeros(eltype(m0),size(m0));
    #x=zeros(eltype(m0),size(m0)); 

 

    #Initial models and parameters that change

    α=α0;
    x = m0;# Initial model
    Jm0=norm(dobs,2)^2;
    Jr0=norm(m0,2)^2;
    J0= Jm0 + Jr0;


   if history
    header = "   k         J              ||y-Ax||²₂             ||x||₁                 ρ                 μ                  ||r||₂                    ||s||₂ "
    println(""); 
    println(""); 
    println(header);
   end


    if ρ isa AbstractFloat
        ρ = fill(ρ, Ne) 
    elseif ρ isa Vector
        ρ = float.(ρ) 
    else
        error("ρ should be either a Float or a Vector.")
    end

    if μ isa AbstractFloat
        μ = fill(μ, Ne)  # Create a vector of size Ni
    elseif μ isa Vector
        μ = float.(μ)  # Ensure it's Float64
    else
        error("μ should be either a Float or a Vector.")
    end
    

    k=0; #counter

    #Main loop

    
    while k < Ne

        k += 1  # Increment iteration counter
        
        b= dobs .- LinearOperator(z .- u ,operators, parameters, adj=false);
        w, Jls= SteepestDescent(x,b,operators, parameters; ρ=ρ[k], Ni=Ni, tol=tolin, α0=α, history=false)
        
        x=  w .+ (z .- u);
        x_old=x;
        
        z_old = z;  # Store previous z for dual residual
        z= SoftThresholding.( x .+ u, ρ[k] , μ[k]); # z-update
          
        u= u .+ (x .- z);

        prim_res = x - z; #Primal residual
        dual_res = ρ[k]*(z - z_old); # Dual residual
        
        
        dpred= LinearOperator(z,operators,parameters,adj=false); #predicted data at iteration k
        Jmk= norm(dobs .- dpred,2)^2;  #||Axₖ - y||₂²
        Jrk= sum(abs.(z));  #||zₖ||₁ at
        Jk = Jmk + μ[k]*Jrk; #objective function at iteration k

        m_diff = norm(x - x_old, 2) # model norm difference 
        res_norm = norm(Jmk, 2) #norma del residuo
        prim_res_norm= norm(prim_res, 2)
        dual_res_norm= norm(dual_res, 2)

        push!(prim_res_history, prim_res_norm)
        push!(dual_res_history, dual_res_norm)
        push!(Jm_history, Jmk); #save misfit at each iteration
        push!(Jr_history, Jrk); #save regularization at each iteration
        push!(J_history,Jk); #save objective
        push!(m_diff_history, m_diff)


        if history && k+1 <= Ne
            println("---------------------------------------------------------------------------------------------------------------------------------------------------------"); 
            @printf("%3.0f %20.10e %20.10e %20.10e  %20.10e %20.10e %20.10e %20.10e\n",k,Jk,Jmk,Jrk,ρ[k],μ[k],prim_res_norm,dual_res_norm);
        end


        if res_norm < tolin

            if history
                println("CGLS converged at iteration $k with residual norm: $res_norm.")
            end
            break
        end

        if length(J_history) > 1 && J_history != 0.0;

            ΔJ= abs((J_history[end] - J_history[end-1]) / J_history[end])

            if round(ΔJ,digits=8) < tolout
               println("Outer-loop for ADMM stopped at $k iterations.")
               println("REASON: ")
               println(" ΔJ = $ΔJ   is < than the established outer loop tolerance = $tolout used.")
               break
            end			
        end
        
    

    
    end


       #possible outputs
       
        #misfit_iteration= res_norm; # residual norm or misfit => ensure you fit the data
        #J_history; objective function 
        #Jm_history; # data fidelity term for χ² test.
        #Jr_history; # model norm fidelity term for χ² test.
        #prim_res_history; primal variable residual
        #dual_res_history; dual variable residual
        #m_diff_history; model difference 


    return x, J_history, Jm_history, Jr_history, get_time

end

