

function IRLS(m0,dobs,operators,parameters; μ=0.5, ϵ=0.01, Ni=100,Ne=10, tol=1e-4, history=true)



	Jirls= Float64[]
	Jm_history= Float64[]
	Jr_history= Float64[]

	weights = ones(Float64,size(m0));

	push!(operators, WeightingOp) #add the weigthing operator to the operator function.
	weight_param= Dict(:w => weights);
	push!(parameters,weight_param )



    if history
        header = "k         ||y-Ax||²₂              ||x||²₂                   μ                   J"
        println(""); 
        println(header);
    end

	m = copy(m0) ;
	k=0;
	


	
	while k < Ne
		
		k+=1
		
		println("iteration $k")
		x, Jls=CGLS(m,dobs,operators, parameters; μ=μ, Ni=Ni, tol=tol, history=false)
	
		m=weights.*x; #
		
		#compute new weights and update operators;
		weights= 1.0 ./( abs.(m).+ ϵ)
		parameters[end][:w] = weights;


		r = dobs - LinearOperator(m, operators, parameters, adj=false) # Update residual
		Jmk= norm(r)^2; #New misfit
        Jrk= norm(m)^2; #New model norm
        Jk = Jmk +μ*Jrk; # New cost function value

        push!(Jirls,Jk); #save objective at each iteration
		push!(Jm_history, Jmk); #save misfit at each iteration
        push!(Jr_history, Jrk); #save regularization at each iteration



        #Tolerance cheking and printing


      if history && k <Ni
            @printf("%3.0f %20.10e %20.10e  %20.10e %20.10e\n", k, Jmk, Jrk, μ, Jk)
      end        

        if length(Jirls) > 1 && Jirls[end] > eps()

            ΔJ= abs((Jirls[end] - Jirls[end-1]) / Jirls[end])

            if round(ΔJ,digits=8) < tol
               println("Loop for CGLS stopped at $k iterations.")
               println("REASON: ")
               println(" ΔJ = $ΔJ   is < than the established tolerance = $tol used.")
               break
            end			
        end




	end

	return m, Jirls, Jls

end

