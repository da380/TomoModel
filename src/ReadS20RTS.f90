program ReadS20RTS

  use nrtype
  use module_util
  use module_function
  use module_fourier
  use module_model

  implicit none

  logical(lgt) :: ltmp
  
  character(len=256) :: spherical_model_file,lateral_model_file
  
  integer(i4b) :: narg,io

  real(dp) :: lambda_l, drmax
  
  ! set the normalisation parameters
  call set_parameters


  ! number of input arguments 
  narg = command_argument_count()

  ! if needed, display input info
  if(narg == 0 .or. narg /= 2) then
     print *, 'inputs: [spherical model] [3D model file]'
     stop
  end if



  ! read in the spherical model file
  call get_command_argument(1,spherical_model_file)
  inquire(file=trim(spherical_model_file),exist=ltmp)
  if(.not.ltmp) stop 'can''t find spherical model file'

  open(newunit = io,file=trim(spherical_model_file),action='read',form='formatted')
  call read_model(io)
  close(io)

  ! Set up the spherical harmonic mesh
  lmax = 64
  call set_sh_grid
  call calc_sph_harm

  ! mesh the radial model
  lambda_l = twopi_d*r(nknot)/(lmax+0.5_dp)
  drmax = lambda_l/2.0_dp
  call mesh_model(drmax) 
  call global_points

  ! read in the 3D model
  call get_command_argument(2,lateral_model_file)
  inquire(file=trim(lateral_model_file),exist=ltmp)
  if(.not.ltmp) stop 'can''t find 3D model file'
  call visc_3d_from_vs(lateral_model_file)


  
end program ReadS20RTS