
from ngsolve import *
from netgen.geom2d import SplineGeometry
from numpy import pi

"""
This code corresponds to Example 1 of the [arXiv preprint](https://arxiv.org/abs/2404.13578)

Tested with NGSolve version 6.2.2402

h-Convergence test for transient elastic/fluid problem 
Manufactured solutions on a squared domain
Pure Dirichlet BCs
HDG approximation in terms of stress & velocity
Crank-Nicolson scheme for time discretization
"""

def weaksym(k, h, dt, tend):
    
    # k: polynomial degree
    # h: mesh size
    # dt: time step
    #

    # ********* Model coefficients and parameters ********* #
    t = Parameter(0.0)
    
    #Parmeters for Table1 
    rhoF, muF, lamF = 1, 1, 1e6
    rhoS, muS, lamS = 1, 1, 1
    
    #Parmeters for Table2
    #rhoF, muF, lamF = 1, 1, 1e6
    #rhoS, muS, lamS = 1e3, 1e6, 1e10
    
    #needed for A = C**{-1}
    aS1 = 0.5 / muS
    aS2 = lamS / (4.0 * muS * (lamS + muS))
    
    aF1 = 0.5 / muF
    aF2 = lamF / (4.0 * muF * (lamF + muF))

    # ******* Exact solution in the fluid domain ****** #
    
    exactp = sin(2*pi*x)*sin(2*pi*y)*sin(t) #Pressure
    # Velocity
    exactuF = CoefficientFunction(( sin(2*pi*x)**2*sin( (8/3)*pi*(y + 1)) ,  -(3/2)*sin(4*pi*x) * sin( ((4/3)*pi*(y + 1)) )**2  ))*sin(2*t)

    # Strain
    epsuF_00 = 4*pi*sin(2*t)*cos(2*pi*x)*sin(2*pi*x)*sin((8*pi*(y + 1))/3)
    epsuF_01 = (4*pi*sin(2*t)*sin(2*pi*x)**2*cos((8*pi*(y + 1))/3))/3 - 3*pi*sin(2*t)*cos(4*pi*x)*sin((4*pi*(y + 1))/3)**2
    epsuF_11 = -4*pi*sin(2*t)*sin(4*pi*x)*cos((4*pi*(y + 1))/3)*sin((4*pi*(y + 1))/3)
    epsuF = CoefficientFunction((epsuF_00, epsuF_01, epsuF_01, epsuF_11), dims = (2,2))
    
    #Stress in the fluid
    sigmaExactF = 2*muF*epsuF - exactp*Id(2)

    divSigma1F = 16*muF*pi**2*sin(2*t)*cos(2*pi*x)**2*sin((8*pi*(y + 1))/3) - 2*pi*cos(2*pi*x)*sin(2*pi*y)*sin(t) - 2*muF*((32*pi**2*sin(2*t)*sin(2*pi*x)**2*sin((8*pi*(y + 1))/3))/9 + 8*pi**2*sin(2*t)*cos(4*pi*x)*cos((4*pi*(y + 1))/3)*sin((4*pi*(y + 1))/3)) - 16*muF*pi**2*sin(2*t)*sin(2*pi*x)**2*sin((8*pi*(y + 1))/3)
    divSigma2F = 2*muF*(12*pi**2*sin(2*t)*sin(4*pi*x)*sin((4*pi*(y + 1))/3)**2 + (16*pi**2*sin(2*t)*cos(2*pi*x)*sin(2*pi*x)*cos((8*pi*(y + 1))/3))/3) - 2*pi*cos(2*pi*y)*sin(2*pi*x)*sin(t) - (32*muF*pi**2*sin(2*t)*sin(4*pi*x)*cos((4*pi*(y + 1))/3)**2)/3 + (32*muF*pi**2*sin(2*t)*sin(4*pi*x)*sin((4*pi*(y + 1))/3)**2)/3
    
    #Fluid source term
    sourceF = - CoefficientFunction( (divSigma1F, divSigma2F) ) + 2*rhoF*CoefficientFunction(( sin(2*pi*x)**2*sin( (8/3)*pi*(y + 1)) ,  -(3/2)*sin(4*pi*x) * sin( ((4/3)*pi*(y + 1)) )**2  ))*cos(2*t)
    
    # ******* Exact solution in the solid domain ****** #
    
    #Displacement
    exactdisp = CoefficientFunction(( sin(2*pi*x)**2*sin( (8/3)*pi*(y + 1)) ,  -(3/2)*sin(4*pi*x) * sin( ((4/3)*pi*(y + 1)) )**2  ))*sin(t)**2

    #Srain
    epsuS_11 = 4*pi*cos(2*pi*x)*sin(2*pi*x)*sin(t)**2*sin((8*pi*(y + 1))/3)
    epsuS_12 = (4*pi*sin(2*pi*x)**2*sin(t)**2*cos((8*pi*(y + 1))/3))/3 - 3*pi*cos(4*pi*x)*sin(t)**2*sin((4*pi*(y + 1))/3)**2
    epsuS_22 = -4*pi*sin(4*pi*x)*sin(t)**2*cos((4*pi*(y + 1))/3)*sin((4*pi*(y + 1))/3)
    epsuS = CoefficientFunction((epsuS_11, epsuS_12, epsuS_12, epsuS_22), dims = (2,2))
    
    #Stress in the solid
    sigmaExactS = 2*muS*epsuS + lamS*Trace(epsuS)*Id(2)

    divSigma1S = -(16*muS*pi**2*sin(t)**2*(13*sin(2*pi*x)**2*sin((8*pi*(y + 1))/3) - 9*cos(2*pi*x)**2*sin((8*pi*(y + 1))/3) + 9*cos(4*pi*x)*cos((4*pi*(y + 1))/3)*sin((4*pi*(y + 1))/3)))/9
    divSigma2S = (2*muS*pi**2*sin(4*pi*x)*sin(t)**2*(26*sin((pi*(16*y + 1))/6) + 18))/3
    
    #Solid source term
    sourceS = - CoefficientFunction( (divSigma1S, divSigma2S) ) + 2*rhoS*CoefficientFunction(( sin(2*pi*x)**2*sin( (8/3)*pi*(y + 1)) ,  -(3/2)*sin(4*pi*x) * sin( ((4/3)*pi*(y + 1)) )**2  ))*cos(2*t)
    
    source = sourceS*IfPos( y ,1,0) + sourceF*IfPos(-y ,1,0)
    
    sigmaExact = sigmaExactS*IfPos( y ,1,0) + sigmaExactF*IfPos(-y ,1,0)
    
        
    # ******* Mesh of the domain  ****** #

    def topBottom():
        geometry = SplineGeometry()
        pnts     = [ (0, -1), (1, -1),  (1, 0), (1, 0.5), (0, 0.5), (0, 0)]
        pnums    = [geometry.AppendPoint(*p) for p in pnts]
        # start-point, end-point, boundary-condition, domain on left side, domain on right side:
        lines = [ (pnums[0], pnums[1], "gammaN",  1, 0),
        (pnums[1], pnums[2], "gammaN",  1, 0),
        (pnums[2], pnums[3], "gammaD",  2, 0),
        (pnums[3], pnums[4], "gammaN",  2, 0),
        (pnums[4], pnums[5], "gammaD",  2, 0),
        (pnums[5], pnums[0], "gammaN",  1, 0),
        (pnums[5], pnums[2], "sigma", 2, 1)]
        for p1,p2,bc,left,right in lines:
            geometry.Append( ["line", p1, p2], bc=bc, leftdomain=left, rightdomain=right)
        
        geometry.SetMaterial(1, "fluid")
        geometry.SetMaterial(2, "solid")
        return geometry
        
    mesh = Mesh(topBottom().GenerateMesh (maxh=h))
    #mesh.GetBoundaries()
    
    domain_values_rho = {'fluid': rhoF,  'solid': rhoS}
    values_list_rho = [domain_values_rho[mat] for mat in mesh.GetMaterials()]
    rho = CoefficientFunction(values_list_rho)
    
    domain_values_a1 = {'fluid': aF1,  'solid': aS1}
    values_list_a1 = [domain_values_a1[mat] for mat in mesh.GetMaterials()]
    a1 = CoefficientFunction(values_list_a1)
    
    domain_values_a2 = {'fluid': aF2,  'solid': aS2}
    values_list_a2 = [domain_values_a2[mat] for mat in mesh.GetMaterials()]
    a2 = CoefficientFunction(values_list_a2)
    
    # ********* Finite dimensional spaces ********* #

    S = L2(mesh, order =k)
    W = VectorL2(mesh, order =k+1)
    What = VectorFacetFESpace(mesh, order=k+1, dirichlet="gammaD|gammaN")
    fes = FESpace([S, S, S, W, What])
    
    # ********* test and trial functions for product space ****** #
    sigma1, sigma12, sigma2, u, uhat = fes.TrialFunction()
    tau1, tau12, tau2, v, vhat = fes.TestFunction()
    
    sigma = CoefficientFunction(( sigma1, sigma12, sigma12, sigma2), dims = (2,2) )
    tau   = CoefficientFunction(( tau1,   tau12,   tau12,   tau2),   dims = (2,2) )


    AsigmaS = aS1 * sigma - aS2 * Trace(sigma) *  Id(mesh.dim)
    AsigmaF = aF1 * sigma - aF2 * Trace(sigma) *  Id(mesh.dim)
    
    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    dS = dx(element_boundary=True)
    
    jump_u = u - uhat
    jump_v = v - vhat

    # ********* Bilinear and linear forms ****** #

    a = BilinearForm(fes, condense=True)
    a += (1/dt)*rho*u*v*dx
    a += (1/dt)*InnerProduct(AsigmaS, tau)*dx("solid") 
    a +=    0.5*InnerProduct(AsigmaF,tau)*dx("fluid")
    a +=   0.5*InnerProduct(sigma, grad(v))*dx    - 0.5*InnerProduct(grad(u), tau)*dx
    a += - 0.5*InnerProduct( sigma*n, jump_v)*dS  + 0.5*InnerProduct(jump_u, tau*n)*dS
    a +=   0.5*((k+1)**2/h)*jump_u*jump_v*dS
    a.Assemble()

    inv_A = a.mat.Inverse(freedofs=fes.FreeDofs(coupling=True))
    
    M = BilinearForm(fes)
    M += (1/dt)*rho*u*v*dx
    M += (1/dt)*InnerProduct(AsigmaS, tau)*dx("solid")
    M += - 0.5*InnerProduct(AsigmaF,tau)*dx("fluid")
    M += - 0.5*InnerProduct(sigma, grad(v))*dx   + 0.5*InnerProduct(grad(u), tau)*dx
    M +=   0.5*InnerProduct( sigma*n, jump_v)*dS - 0.5*InnerProduct(jump_u, tau*n)*dS
    M += - 0.5*((k+1)**2/h)*jump_u*jump_v*dS
    M.Assemble()

    ft = LinearForm(fes)
    ft += source * v * dx
    ft += -((sigmaExactF- sigmaExactS)*n)*vhat.Trace()*ds(definedon=mesh.Boundaries("sigma"))
    #ft += (sigmaExact*n)*vhat.Trace()*ds(definedon=mesh.Boundaries("gammaN"))

    # ********* instantiation of initial conditions ****** #
    
    disp0 = GridFunction(W)
    disp0.Set(exactdisp)

    disp = GridFunction(W)
    
    u0 = GridFunction(fes)
    u0.components[0].Set(sigmaExact[0,0])
    u0.components[1].Set(sigmaExact[0,1])
    u0.components[2].Set(sigmaExact[1,1])
    u0.components[3].Set(exactuF)
    u0.components[4].Set(exactuF, dual=True)

    u1 = GridFunction(fes)
    
    ft.Assemble()
    
    res = u0.vec.CreateVector()
    b0  = u0.vec.CreateVector()
    b1  = u0.vec.CreateVector()
    
    b0.data = ft.vec

    t_intermediate = dt # time counter within one block-run
    
    # ********* Time loop ************* #

    while t_intermediate < tend:

        t.Set(t_intermediate)
        ft.Assemble()
        b1.data = ft.vec
     
        res.data = M.mat*u0.vec + 0.5*(b0.data + b1.data)

        u1.vec[:] = 0.0
        u1.components[4].Set(exactuF, BND)#, dual=True)#BDN --->  Dirichlet Boundary

        res.data = res - a.mat * u1.vec
        res.data += a.harmonic_extension_trans * res
        u1.vec.data += inv_A * res
        u1.vec.data += a.inner_solve * res
        u1.vec.data += a.harmonic_extension * u1.vec
        
        disp.vec[:] = 0.0
        disp.vec.data = disp0.vec + 0.5*dt*(u1.components[3].vec + u0.components[3].vec)
        
        u0.vec.data = u1.vec
        disp0.vec.data = disp.vec
        b0.data = b1.data
        t_intermediate += dt
        
        print("\r",t_intermediate,end="")

    # ********* L2-errors at time tend ****** #
    
    gfsigma1, gfsigma12, gfsigma2, gfu = u0.components[0:4]

    gfsigma = CoefficientFunction(( gfsigma1, gfsigma12, gfsigma12, gfsigma2), dims = (2,2) )

    norm_u=  (gfu - exactuF) * (gfu - exactuF)
    norm_u = Integrate(norm_u, mesh)
    norm_u = sqrt(norm_u)
    
    norm_p= (Trace(gfsigma) + (2*muF/lamF + mesh.dim)*exactp) * (Trace(gfsigma) + (2*muF/lamF + mesh.dim)*exactp)*IfPos(-y , 1, 0)
    norm_p = Integrate(norm_p, mesh)
    norm_p = sqrt(norm_p)
    
    norm_d=  (disp - exactdisp) * (disp - exactdisp)*IfPos( y , 1, 0)
    norm_d = Integrate(norm_d, mesh)
    norm_d = sqrt(norm_d)

    norm_s = InnerProduct(a1*(sigmaExact - gfsigma) - a2*Trace(sigmaExact - gfsigma)*Id(mesh.dim), sigmaExact - gfsigma)
    norm_s = Integrate(norm_s, mesh)
    norm_s = sqrt(norm_s)

    return norm_s, norm_u, norm_d, norm_p



# ********* Collect errors ************* #

def collecterrors(k, maxr, dt, tend):
    l2e_s = []
    l2e_v = []
    l2e_d = []
    l2e_p = []
    for l in range(0, maxr):
        hl = 2**(-(l + 1))
        er_1, er_2, er_3, er_4 = weaksym(k, hl, dt, tend)
        l2e_s.append(er_1)
        l2e_v.append(er_2)
        l2e_d.append(er_3)
        l2e_p.append(er_4)
    return l2e_s, l2e_v, l2e_d, l2e_p


# ********* Convergence table ************* #

def hconvergenctauEble(e_1, e_2, e3, e_p, maxr):
    print("==========================================================================================")
    print(" Mesh   Errors_s    Order    Error_u   Order    Error_d   Order   Error_p   Order ")
    print("------------------------------------------------------------------------------------------")
    rate1 = []
    rate2 = []
    rate3 = []
    rate_p = []
    for i in range(maxr):
        rate1.append('  *  ')
        rate2.append('  *  ')
        rate3.append('  *  ')
        rate_p.append('  *  ')
    for i in range(1, maxr):
        if abs(e_1[i]) > 1.e-15 and abs(e_2[i]) > 1.e-15 and abs(e3[i]) > 1.e-15:
            rate1[i] = format(log(e_1[i - 1] / e_1[i]) / log(2), '+5.2f')
            rate2[i] = format(log(e_2[i - 1] / e_2[i]) / log(2), '+5.2f')
            rate3[i] = format(log(e3[i - 1] / e3[i]) / log(2), '+5.2f')
            rate_p[i] = format(log(e_p[i - 1] / e_p[i]) / log(2), '+5.2f')
    for i in range(maxr):
        print(" 1/%-4d %8.2e   %s   %8.2e   %s   %8.2e   %s  %8.2e   %s  " %
              (2**(i + 1), e_1[i], rate1[i],
               e_2[i], rate2[i], e3[i], rate3[i], e_p[i], rate_p[i]))

    print("==========================================================================================")



# ********* MAIN DRIVER ************* #

maxlevels = 6 # Levels of mesh refinement
p = 1 # Polynomial degree
dt = (1/32)**(p/2+1) # Time step
tend = 0.3 # Final time

er_s, er_v, er_d, er_p = collecterrors(p, maxlevels, dt, tend)
hconvergenctauEble(er_s, er_v, er_d, er_p, maxlevels)
