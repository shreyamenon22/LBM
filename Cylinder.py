import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

Ni= 15000 #iterations
Reynolds_no=80
N_x=300
N_y=50
Cylinder_cx= N_x //5
Cylinder_cy=N_y//2
Cylinder_r=N_y//9

Max_horizontal_iv=0.04
VISUALIZE=True
Plot_Every_N_Step=100
Skip_first_N_Iteration=0

D_v=9

Lattice_velocities=jnp.array([ [ 0,  1,  0, -1,  0,  1, -1, -1,  1,],[ 0,  0,  1,  0, -1,  1,  1, -1, -1,]])
Lattice_indices=jnp.array([0,1,2,3,4,5,6,7,8])
opp_lattices_indices=jnp.array([0,3,4,1,2,7,8,5,6])

Lattice_weights=jnp.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36]) 

U_right=jnp.array([1,5,8])
U_up=jnp.array([2,5,6])
U_left=jnp.array([3,6,7])
U_down=jnp.array([4,7,8])
U_pure_vertical=jnp.array([0,2,4])
U_pure_horizontal=jnp.array([0,1,3])

def get_density(discrete_velocity):
    density=jnp.sum(discrete_velocity,axis=-1)#summing over all     9 discrete velocities(last axis)
    return density

def get_macroscopic_velocities(discrete_velocity,density):#(discrete velocity * lattice velocity)/density
    macroscopic_velocities=jnp.einsum(
        "NMQ,dQ->NMd",discrete_velocity,Lattice_velocities
    )/density[...,jnp.newaxis]
    return macroscopic_velocities

def get_equilibrium_discrete_velocities(macroscopic_velocities,density):
    projected_discrete_velocity= jnp.einsum(
        "dQ,NMd->NMQ"
        ,Lattice_velocities,
        macroscopic_velocities
    ) #projected macroscopic on lattice vel
    macroscopic_velocity_magnitude=jnp.linalg.norm(
        macroscopic_velocities,
        axis=-1,#over last axis
        ord=2#eucledion norm
    )
    equilibrium_discrete_velocities= (
        density[...,jnp.newaxis]
        *
        Lattice_weights[jnp.newaxis,jnp.newaxis,:]
        *
        (
            1+3*projected_discrete_velocity+9/2*projected_discrete_velocity**2-3/2*macroscopic_velocity_magnitude[...,jnp.newaxis]**2

        )
    )
    return equilibrium_discrete_velocities


def main():
    jax.config.update("jax_enable_x64", True)
    kinematic_viscosity=(
        (
            Max_horizontal_iv*Cylinder_r
        )/(Reynolds_no)
    )
    relaxation_omega=(
        (1.0)/(3.0*kinematic_viscosity+0.5)
      )
    #define a mesh
    x=jnp.arange(N_x)
    y=jnp.arange(N_y)
    X,Y=jnp.meshgrid(x,y,indexing="ij")

    #obstacle Mask :array of shape like x y but true is point belongs to obstacle false if not
    obstacle_mask=(
        jnp.sqrt(
            (X-Cylinder_cx)**2+(Y-Cylinder_cy)**2)<Cylinder_r
    )
    velocity_profile = jnp.zeros((N_x,N_y,2))
    velocity_profile=velocity_profile.at[:,:,0].set(Max_horizontal_iv)
    
    @jax.jit
    def update(discrete_velocities_prev):
        #prescribe outflow BC on R boundary
        discrete_velocities_prev=discrete_velocities_prev.at[-1,:,U_left].set(
            discrete_velocities_prev[-2,:,U_left]
        )
        #Macroscopic Velocities
        density_prev=get_density(discrete_velocities_prev)
        macroscopic_velocities_prev= get_macroscopic_velocities(
            discrete_velocities_prev,density_prev
        )
        # Prescribe Inflow Dirichlet BC using Zou/He scheme
        macroscopic_velocities_prev=\
            macroscopic_velocities_prev.at[0,1:-1,:].set(velocity_profile[0,1:-1,:])
        density_prev=density_prev.at[0,:].set(
    (
        get_density(discrete_velocities_prev[0,:,U_pure_vertical].T)
        +2*get_density(discrete_velocities_prev[0,:,U_left].T)
    )/(1-macroscopic_velocities_prev[0,:,0])
)
        #compute discrete eq vel
        equilibrium_discrete_velocities=get_equilibrium_discrete_velocities(macroscopic_velocities_prev,density_prev)

        #belongs to Zou/He scheme
        discrete_velocities_prev=\
        discrete_velocities_prev.at[0,:,U_right].set(
            equilibrium_discrete_velocities[0,:,U_right]
        )

        #collide acc to BGK
        discrete_velocities_post_collision=(
            discrete_velocities_prev-relaxation_omega*(discrete_velocities_prev-equilibrium_discrete_velocities )
        )
        #bounceback boundary condition to enforce no-slip
        for i in range(D_v):
            discrete_velocities_post_collision=\
                discrete_velocities_post_collision.at[obstacle_mask,Lattice_indices[i]].set(
                    discrete_velocities_prev[obstacle_mask,opp_lattices_indices[i]]
                )
        #stream along lattice vel
        discrete_velocities_streamed=discrete_velocities_post_collision
        for i in range(D_v):
            discrete_velocities_streamed=discrete_velocities_streamed.at[:,:,i].set(
                jnp.roll(
                    jnp.roll(discrete_velocities_post_collision[:,:,i],
                             Lattice_velocities[0,i],
                             axis=0
                             ),
                             Lattice_velocities[1,i],
                             axis=1,
                )
            )
        return discrete_velocities_streamed 
    
    discrete_velocities_prev = get_equilibrium_discrete_velocities(
    velocity_profile,
    jnp.ones((N_x, N_y))  
    )

    plt.style.use("dark_background")
    plt.figure(figsize=(15,6),dpi=100)


    for i in tqdm(range(Ni)):
        discrete_velocities_next = update(discrete_velocities_prev)
        discrete_velocities_prev = discrete_velocities_next

        if i % Plot_Every_N_Step == 0 and VISUALIZE and i > Skip_first_N_Iteration:  
            density = get_density(discrete_velocities_next)
            macroscopic_velocities = get_macroscopic_velocities(
                discrete_velocities_next,
                density
            )
            velocity_mag = jnp.linalg.norm(
                macroscopic_velocities,
                axis=-1,
                ord=2,
            )
            d_u__dx,du__dy=jnp.gradient(macroscopic_velocities[...,0])
            dv__dx,dv_dy=jnp.gradient(macroscopic_velocities[...,1])
            curl=(du__dy-dv__dx)

            #vel mag contour plot in top
            plt.subplot(211)
            plt.contourf(
                X,
                Y,
                velocity_mag,
                levels=50,
                cmap=cmr.amber
            )
            plt.colorbar().set_label("Velocity Magnitude")
            plt.gca().add_patch(plt.Circle(
                (Cylinder_cx,Cylinder_cy),Cylinder_r,
            ))
            # voroicity mag Contour Plot in botton
            plt.subplot(212)
            plt.contourf(
                X,
                Y,
                curl,
                levels=50,
                cmap=cmr.redshift,
                vmin=-0.02,
                vmax=0.02,)
            plt.colorbar().set_label("Vorticity Magnitude")
            plt.gca().add_patch(plt.Circle(
                (Cylinder_cx,Cylinder_cy),
                Cylinder_r,
                color="darkblue",
            ))
            plt.draw()
            plt.pause(0.0001)
            plt.clf()
    if VISUALIZE:
        plt.show()


if __name__=="__main__":
    main()
        