
from __future__ import division
from lib import *

class Grating(stim(width=200.0,fs=10.0,ph=0.0,speed=0.0,contr=1.0,
                   box=False,Lbg=24.65,Lmin=0.0,Lmax=49.3,gamma=2.5,BTRR=63.2)):
    """Grating stimulus
    
    :param width: size of stimulus in pixels
    :param fs: spatial frequncy
    :param ph: phase
    :param speed: speed
    :param contr: contrast  
    """
    frag_source = """
        precision highp float;
        
        uniform float fs;
        uniform float phase;
        uniform float contr;
        uniform float box;
        uniform float Lbg, Lmin, Lmax, gamma, BTRR;

        vec3 lum2image(float c) {
            vec3 vout;
            float U = 255.0*pow(((1+c)*Lbg-Lmin)/(Lmax-Lmin),1.0/gamma);
            float b1 = (BTRR+1.0)/BTRR;
            //float b = min(floor(U*b1),255.0);
            float b = min(floor(U*b1),255.0) + 1.0/32.0;
            //float b2 = floor((U-b/b1)*(BTRR+1.0));
            float b2 = floor((U-b/b1)*(BTRR+1.0)) + 1.0/32.0;
            vout = vec3(b2/255.0, 0.0 ,b/255.0);
            return vout;
        }

        void main() {
            float x = gl_TexCoord[0].x - 0.5;
            float y = -(gl_TexCoord[0].y - 0.5);
            float m = exp(-0.5*(x*x + y*y)/pow(0.17,2.0));
            float c = m*contr*cos(fs*x + phase);
            if (box) {
                gl_FragColor.rgba = vec4(lum2image(c),1.0);
            } else {
                c = (1.0+c)/2.0;
                gl_FragColor.rgba = vec4(c,c,c,1.0);
            }
        }
        """
    def __init__(self,pos,params=Params()):
        self.pos = pos
        self.params = copy_params(self._defaults,params)
        self.clock = pyglet.clock.Clock()
        self.t0 = self.clock.time()
        self.shader = Shader(self.frag_source)
        self.program = self.shader.program
        self.uniforms = dict(map(self.shader.uniform,
                                 ['fs','phase','contr','box','Lbg','Lmin','Lmax','gamma','BTRR']))
        glUseProgram(self.program)
        glUniform1f(self.uniforms['phase'],0.0)
        glUniform1f(self.uniforms['fs'],self.params.fs)
        glUniform1f(self.uniforms['contr'],self.params.contr)
        glUniform1f(self.uniforms['box'],self.params.box)
        glUniform1f(self.uniforms['Lbg'],self.params.Lbg)
        glUniform1f(self.uniforms['Lmin'],self.params.Lmin)
        glUniform1f(self.uniforms['Lmax'],self.params.Lmax)
        glUniform1f(self.uniforms['gamma'],self.params.gamma)
        glUniform1f(self.uniforms['BTRR'],self.params.BTRR)
        print self.params.box
        glUseProgram(0)
    
    def draw(self):
        p = self.params
        x, y = self.pos
        w2 = p.width/2.0
        glUseProgram(self.program)
        ph = p.speed*(self.clock.time()-self.t0)
        glUniform1f(self.uniforms['phase'],ph)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0],self.pos[1],0.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0,1.0)
        glVertex2f(-w2,-w2)
        glTexCoord2f(1.0,1.0)
        glVertex2f(w2,-w2)
        glTexCoord2f(1.0,0.0)
        glVertex2f(w2,w2)
        glTexCoord2f(0.0,0.0)
        glVertex2f(-w2,w2)
        glEnd()
        glPopMatrix()
        glUseProgram(0)

class Rectangle(stim(width=200.0,height=200.0,theta=0.0)):
    vert_source = """
        uniform float angle;
        
        void main() {
            vec4 a = gl_Vertex;
            vec4 b = a;
            b.y = a.y*cos(angle) - a.z*sin(angle);
            b.z = a.z*cos(angle) + a.y*sin(angle);

            gl_TexCoord[0] = gl_MultiTexCoord0;
            gl_Position = gl_ModelViewProjectionMatrix*b;
        }"""
    frag_source = """
        uniform float fs;
        uniform float phase;

        void main() {
            float x = gl_TexCoord[0].x - 0.5;
            float y = -(gl_TexCoord[0].y - 0.5);
            float c = (1.0 + cos(fs*x + phase))/2.0;
            float m = exp(-0.5*(x*x + y*y)/pow(0.17,2.0));
            gl_FragColor.rgba = vec4(c,c,c,m);
        }
        """
    def __init__(self,pos,params=Params()):
        self.pos = pos
        self.params = copy_params(self._defaults,params)
        self.clock = pyglet.clock.Clock()
        self.t0 = self.clock.time()
        self.shader = Shader(self.frag_source, self.vert_source)
        self.program = self.shader.program
        self.uniforms = dict(map(self.shader.uniform,['fs','phase','angle','w','h']))
        glUseProgram(self.program)
        glUniform1f(self.uniforms['phase'],0.0)
        glUniform1f(self.uniforms['fs'],20.0)
        glUniform1f(self.uniforms['angle'],0.2)
        glUseProgram(0)
    def draw(self):
        glUseProgram(self.program)
        ph = 40.0*(self.clock.time()-self.t0)
        glUniform1f(self.uniforms['phase'],ph)
        glUniform1f(self.uniforms['angle'],ph/40.0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0],self.pos[1],0.0)
        w2 = self.params.width/2
        glColor3f(255.0,0,0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0,1.0)
        glVertex2f(-w2,-w2)
        glTexCoord2f(1.0,1.0)
        glVertex2f(w2,-w2)
        glTexCoord2f(1.0,0.0)
        glVertex2f(w2,w2)
        glTexCoord2f(0.0,0.0)
        glVertex2f(-w2,w2)
        glEnd()
        glUseProgram(0)
        glPopMatrix()

class Line(stim(length=100.0,width=10.0,angle=0.0,color=(1.0,1.0,1.0,1.0))):
    def __init__(self,pos,params=Params()):
        self.pos = pos
        if hasattr(params,'color') and not iterable(params.color):
            params.color = [params.color]*3
        self.params = copy_params(self._defaults,params)
        p = self.params
        self.clock = pyglet.clock.Clock()
        self.t0 = self.clock.time()
        # make indexed vertex_list
        ind = array([0,1,2,0,2,3])
        xy = r_[0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0].reshape(-1,2) - 0.5
        self.vlist = pyglet.graphics.vertex_list_indexed(4, ind, 'v2d/stream', 'c4f')
        N = 4
        self.xy = from_ctypes(self.vlist.vertices,'f8',(N,2))
        self.xy[...] = xy * r_[p.length,p.width]
        self.colors = from_ctypes(self.vlist.colors,'f4',(N, 4))
        self.colors[:,3] = 1.0
        self.colors[:,:len(p.color)] = p.color
        self.vlist._vertices_cache.invalidate()
        self.vlist._colors_cache.invalidate()
    
    def draw(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0],self.pos[1],0.0)
        glRotatef(self.params.angle,0.0,0.0,1.0)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_MULTISAMPLE_ARB)
        #glEnable(GL_POLYGON_SMOOTH);
        #glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        self.vlist.draw(GL_TRIANGLES)
        glEnable(GL_TEXTURE_2D)
        #glDisable(GL_POLYGON_SMOOTH);
        #glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE)
        glDisable(GL_MULTISAMPLE_ARB)
        glPopMatrix()

class Movie(stim(fname=None,fps=60.0,scale=1.0,rotation=0.0)):
    def __init__(self,pos,params=Params()):
        self.pos = pos
        self.clock = pyglet.clock.Clock()
        self.params = copy_params(self._defaults,params)
        self.batch = pyglet.graphics.Batch()
        self.set_fname(self.params.fname)
    
    def set_fname(self,fname):
        im = None
        if fname:
            self.params.fname = fname
            if fname.endswith('.mat'):
                from scipy.io import loadmat
                frames = loadmat()['frames']
                h, w = frames.shape[:2]
                frames = iter(frames)
            else:
                from movieiter import ffmpegsrc
                frames, w, h = ffmpegsrc(os.path.normpath(fname))
            frm = frames.next()
            im = pyglet.image.ImageData(width=w,height=h,format='RGB',data=frm,pitch=-w*3)
            self.cframe = 0
            im.anchor_x = w//2
            im.anchor_y = h//2
            self.spr = pyglet.sprite.Sprite(im,batch=self.batch)
            self.spr.position = (self.pos[0],self.pos[1])
            self.spr.scale = self.params.scale
            self.spr.rotation = self.params.rotation
            self.frames = frames
        
        self.image = im

    def reset(self):
        self.frames, w, h = ffmpegsrc(self.params.fname)
    
    def draw(self):
        self.t0 = self.clock.time()
        self.draw_next()
        self.draw = self.draw_next
    
    def draw_next(self):
        glColor4f(255.0,255.0,255.0,255.0)
        if (self.clock.time() - self.t0) > 1.0/self.params.fps:
            self.image.set_data('RGB',-self.image.width*3,self.frames.next())
            self.t0 = self.clock.time()
        if self.image:
            cx, cy = self.pos
            self.spr._group.texture = self.image.get_texture()
            self.spr.draw()

class MovieGrid(stim(fname='movie/fft.mp4',fps=29.97,
                     tilt=0.0, slant=0.0,
                     tile_image_coords=[makerect(0,0,0.5,1.0),makerect(0.5,0,0.5,1.0)],
                     tile_disp_coords=[makecrect(-200,0,150,120),makerect(200,0,150,120)],
                     tile_disp_anchors=[(-200,0,0),(200,0,20)],
                     tile_slants=[5,10], tile_tilts=[10,-10], tile_colors=[0.5,1],
                     tile_alphas=[1.0,1.0],mask='')):
    def __init__(self,pos,params):
        self.params = copy_params(self._defaults)
        self.params.__dict__.update(params.__dict__)
        p = self.params
        # load movie
        fname = p.fname
        from movieiter import ffmpegsrc
        if not os.path.exists(fname):
            raise Exception('Movie file "%s" does not exist!'%fname)
        frames, w, h = ffmpegsrc(fname)
        frm = frames.next()
        im = pyglet.image.ImageData(width=w,height=h,format='RGB',data=frm,pitch=-w)
        self.tex = im.get_texture()
        rx, ry = self.tex.tex_coords[6:8]
        p.tile_image_coords = array(p.tile_image_coords).reshape(-1,4,2)*r_[rx,ry]
        p.tile_disp_coords = array(p.tile_disp_coords).reshape(-1,4,3)
        p.tile_disp_anchors = array(p.tile_disp_anchors).reshape(-1,3)
        self.frames = frames
        self.clock = pyglet.clock.Clock()
        self.t0 = self.clock.time()
        # initialize
        N = p.tile_disp_anchors.shape[0]
        if not hasattr(p.tile_slants,'__len__'): p.tile_slants = [p.tile_slants]
        if not hasattr(p.tile_tilts,'__len__'): p.tile_tilts = [p.tile_tilts]
        if not hasattr(p.tile_colors,'__len__'): p.tile_colors = [p.tile_colors]
        p.tile_colors = reshape(p.tile_colors,(N,-1))
        if not hasattr(p.tile_alphas,'__len__'): p.tile_alphas = [p.tile_alphas]
        p.tile_alphas = reshape(p.tile_alphas,(N,-1))
        # check for shape time-series
        if hasattr(p,'series'):
            from itertools import cycle
            Nser = p.series
            imser = array(p.tile_image_coords).reshape(Nser,-1,4,2)
            dsser = array(p.tile_disp_coords).reshape(Nser,-1,4,3)
            p.tile_image_series = [ cycle(imser[i]) for i in xrange(imser.shape[0]) ]
            p.tile_disp_series = [ cycle(dsser[i]) for i in xrange(dsser.shape[0]) ]
        self.batch, self.vlist, self.indices = make_indarray(N)
        self.xy = from_ctypes(self.vlist.vertices,'f8',(4*N,3))
        self.colors = from_ctypes(self.vlist.colors,'f4',(N,4,4))
        self.txy = from_ctypes(self.vlist.tex_coords,'f4',(4*N,2))
        # fill the arrays
        self.colors[...] = 1.0
        for i in xrange(N):
            self.colors[i,:,:3] = p.tile_colors[i]
            self.colors[i,:,3] = p.tile_alphas[i]
        #self.colors[:,:,:3] = self.params.color
        for i in xrange(N):
            self.xy[i*4:(i+1)*4,:] = p.tile_disp_coords[i]
            self.txy[i*4:(i+1)*4,:] = p.tile_image_coords[i]
        # initialize mask
        self.set_mask(p.mask)
        # 
        self.im = im
        self.N = N
        self.pos = pos
        # transform user-specified slants and tils from degrees to rads
        p.tile_slants = pi*array(p.tile_slants)/180.0
        p.tile_tilts = pi*array(p.tile_tilts)/180.0
        # transform each tile
        for i in xrange(self.N):
            xy = slanttilt(p.tile_disp_coords[i], p.tile_slants[i], p.tile_tilts[i], p.tile_disp_anchors[i])
            self.xy[i*4:(i+1)*4,:] = xy
        self.vlist._vertices_cache.invalidate()

    frag_source = """
        uniform sampler2D tex1;
        uniform sampler2D mask;
        uniform int usemask;
        uniform vec2 pixel;
        uniform vec2 offset;
        
        void main() {
            vec4 color1 = texture2D(tex1,gl_TexCoord[0].st);
            if (usemask == 1) {
                vec4 color2 = texture2D(mask,(gl_TexCoord[0].st-offset)*pixel);
                gl_FragColor.rgba = vec4(color1.r,color1.g,color1.b,color2.r);
            } else if (usemask < 0) {
                gl_FragColor.rgba = vec4(color1.r,color1.g,color1.b,color1.a);
            } else {
                gl_FragColor.rgba = vec4(color1.r,color1.g,color1.b,1.0);
            }
        }
        """
    def setup_shader(self):
        self.shader = Shader(self.frag_source)
        self.program = self.shader.program
        self.uniforms = dict(map(self.shader.uniform, ['tex1','mask','usemask','pixel','offset']))
        #rix, riy = self.tex.tex_coords[6:8]
        rix = (self.txy[1,0] - self.txy[0,0])#*rix*rix
        riy = (self.txy[2,1] - self.txy[1,1])
        #ax = (self.txy[1,0] - self.txy[0,0])/rix
        #ay = (self.txy[2,1] - self.txy[1,1])/riy
        #rx, ry = (ax*ax)/self.tex.width, (ay*ay)/self.tex.height
        #rix, riy = ax, ay
        glUseProgram(self.program)
        glUniform1i(self.uniforms['tex1'],0)
        if self.mask == '':
            glUniform1i(self.uniforms['usemask'],0)
        elif self.mask == 'alpha':
            glUniform1i(self.uniforms['usemask'],-1)
        else:
            rmx, rmy = self.mask.tex_coords[6:8]
            #mx, my = self.mask.width, self.mask.height
            glUniform1i(self.uniforms['usemask'],1)
            glUniform1i(self.uniforms['mask'],1)
            glUniform2f(self.uniforms['pixel'],float(rmx)/rix,float(rmy)/riy)
            glUniform2f(self.uniforms['offset'],self.txy[0,0],self.txy[0,1])
            
        glUseProgram(0)
        
    def set_mask(self,fname):
        self.mask = fname
        if fname:
            if fname == 'alpha':
                self.mask = 'alpha'
            else:
                self.mask = pyglet.image.load(fname).get_texture()
        self.setup_shader()
    
    def _frmupdate(self,t):
        if (self.clock.time() - self.t0) > 1.0/self.params.fps:
            self.im.set_data('RGB',-self.im.width*3,self.frames.next())
            self.tex = self.im.get_texture()
            self.t0 = self.clock.time()
            
            if hasattr(self.params,'series'):
                p = self.params
                for i in xrange(len(p.tile_image_series)):
                    self.txy[i*4:(i+1)*4,:] = p.tile_image_series[i].next().copy()
                    xy = slanttilt(p.tile_disp_series[i].next(), p.tile_slants[i], p.tile_tilts[i], p.tile_disp_anchors[i])
                    self.xy[i*4:(i+1)*4,:] = xy
                self.vlist._vertices_cache.invalidate()
                self.vlist._tex_coords_cache.invalidate()
    
    def draw(self):
        self._frmupdate(self.clock.time())
        p = self.params
        # ready to draw
        glPushMatrix()
        glLoadIdentity()
        #glEnable(GL_DEPTH_TEST)
        glTranslatef(self.pos[0],self.pos[1],0.0)
        glRotatef(-p.slant,1.0,0.0,0.0)
        glRotatef(p.tilt,0.0,1.0,0.0)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D,self.tex.id)
        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.tex.id)
        if not isinstance(self.mask,str):
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D,self.mask.id)
        self.vlist.draw(GL_TRIANGLES)
        glUseProgram(0)
        glDisable(GL_TEXTURE_2D)
        #glDisable(GL_DEPTH_TEST)
        glPopMatrix()

class MaskNumpy(stim(width=500.0,height=500.0,sigma=200.0,edge=2.0)):
    """Mask stimulus computed using numpy
    
    :param sigma: mask radius
    :param edge: power of apperture's Gaussian (slope)
    """
    def __init__(self,pos,params=Params()):
        self.params = copy_params(self._defaults,params)
        p = self.params
        self.pos = pos      
        # and (x,y) coordinate for each dot
        self.mask = make_mask(int(p.width),int(p.height),p.sigma,p.edge)
    
    def draw(self):
        p = self.params
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0],self.pos[1],0.0)
        self.mask.draw()
        glPopMatrix()

class Mask(stim(width=500.0,height=500.0,bgcolor=(0.0,0.0,0.0),sigma=200.0,edge=2.0)):
    """Mask stimulus (GLSL)
    
    :param sigma: mask radius
    :param edge: power of apperture's Gaussian (slope)
    """
    frag_source = """
        uniform vec3 bgcolor;
        uniform float sigma, edge;    
        void main() {
            float x = gl_TexCoord[0].x - 0.5;
            float y = -(gl_TexCoord[0].y - 0.5);
            float m = exp(-0.5*pow(x*x + y*y,edge/2.0)/pow(sigma,edge));
            gl_FragColor.rgba = vec4(bgcolor,1.0-m);
        }
        """
    def __init__(self,pos,params=Params()):
        self.pos = pos
        self.params = copy_params(self._defaults,params)
        p = self.params
        self.shader = Shader(self.frag_source)
        self.program = self.shader.program
        self.uniforms = dict(map(self.shader.uniform,['bgcolor','sigma','edge']))
        glUseProgram(self.program)
        glUniform3f(self.uniforms['bgcolor'],*p.bgcolor)
        glUniform1f(self.uniforms['sigma'],p.sigma)
        glUniform1f(self.uniforms['edge'],p.edge)
        glUseProgram(0)
    
    def draw(self):
        glUseProgram(self.program)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0],self.pos[1],0.0)
        w2 = self.params.width/2
        h2 = self.params.height/2
        glBegin(GL_QUADS)
        glTexCoord2f(0.0,1.0)
        glVertex2f(-w2,-h2)
        glTexCoord2f(1.0,1.0)
        glVertex2f(w2,-h2)
        glTexCoord2f(1.0,0.0)
        glVertex2f(w2,h2)
        glTexCoord2f(0.0,0.0)
        glVertex2f(-w2,h2)
        glEnd()
        glUseProgram(0)
        glPopMatrix()


DotLatticeBase = stim(ap_fs=0.0, ap_sigma=100.0, ap_edge=4.0,
                      gamma=70.0, theta=10.0, dx=40.0, r=1.5,
                      dot_size=150.0, dot_sigma=0.25, dot_fs=0.0,
                      dot_edge=10.0, dot_c=0.4, dot_phi=0.2)

class DotLattice(DotLatticeBase):
    """Dot-lattice stimulus
    
    :param ap_fs: spatial frequency of the mask
    :param ap_sigma: spatial frequency of the mask
    :param ap_edge: power of apperture's Gaussian (slope)
    :param gamma: angle 
    :param theta: angle of A nd B directions within lattice
    :param dx: parameter a, shortest distance
    :param r: aspect ratio
    :param dot_size: size of dot's texture
    :param dot_sigma: width of dot's Gaussian
    :param dot_fs: spatial frequency of dot's grating
    :param dot_edge: power of dot's Gaussian (slope)
    :param dot_c: dot color [0.0,1.0]
    :param dot_ph: phase of dot's grating
    """
    def __init__(self,pos,params=Params()):
        self.params = copy_params(self._defaults,params)
        p = self.params
        self.pos = pos      
        # prepare image of dot of the lattice
        self.dot = make_dot(p.dot_sigma,p.dot_fs,p.dot_phi,p.dot_edge).get_texture()
        # and (x,y) coordinate for each dot
        lw = xg(0.01,p.ap_sigma,p.ap_edge) # "xg" is a helper that return x at known y of Gaussian
        # genrate dot center coordintes within the circle of radius lw
        xy = make_lattice(lw,p.dx,p.r,p.gamma,p.theta)
        
        # to draw thw dots we will use OpenGL DrawArray through pyglet's vertex_list
        N = xy.shape[0]
        self.vlist = pyglet.graphics.vertex_list(N, 'v2d/stream', 'c4f')
        self.xy = from_ctypes(self.vlist.vertices,'f8',(N,2))
        self.xy[...] = xy
        self.colors = from_ctypes(self.vlist.colors,'f4',(N, 4))
        self.colors[:,:] = p.dot_c
        self.colors[:,3] = 1.0
        # prepare the mask (apperture)
        w = 2*int(1.1*abs(self.xy).max())
        self.mask = make_mask(w,w,p.ap_sigma,p.ap_edge)
        # store the parameter in this class 
    
    def draw(self):
        p = self.params
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0],self.pos[1],0.0)
        glRotatef(p.gamma,0.0,0.0,1.0)
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_TEXTURE_2D)
        glPointSize(p.dot_size)
        glBindTexture(GL_TEXTURE_2D,self.dot.id)
        self.vlist.draw(pyglet.gl.GL_POINTS)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_POINT_SPRITE)
        # draw the mask (apperture)
        self.mask.draw()
        glPopMatrix()

class Lattice(stim(width=400.0,height=400.0,
                   gamma=70.0, theta=10.0, dx=40.0, r=1.5,
                   dot_size=150.0, dot_sigma=0.25, dot_fs=0.0,
                   dot_edge=10.0, dot_c=0.4, dot_phi=0.2,
                   slant=0.0,tilt=0.0)):
    """Dot-lattice stimulus
    
    :param width: width of stimulus in pixels
    :param height: height of stimulus in pixels
    :param gamma: angle 
    :param theta: angle
    :param dx: parameter a, shortest distance
    :param r: aspect ratio
    :param dot_size: size of dot's texture
    :param dot_sigma: width of dot's Gaussian
    :param dot_fs: spatial frequency of dot's grating
    :param dot_edge: power of dot's Gaussian (slope)
    :param dot_c: dot contrast [0.0,1.0]
    :param dot_ph: phase of dot's grating
    """
    def __init__(self,pos,params=Params()):
        self.params = copy_params(self._defaults,params)
        p = self.params
        self.pos = pos      
        # prepare image of dot of the lattice
        self.dot = make_dot(p.dot_sigma,p.dot_fs,p.dot_phi,p.dot_edge).get_texture()
        # and (x,y) coordinate for each dot
        xy = make_lattice(p.width/2.0,p.dx,p.r,p.gamma,p.theta)
        # to draw the dots we will use OpenGL DrawArray through pyglet's vertex_list
        N = xy.shape[0]
        self.vlist = pyglet.graphics.vertex_list(N, 'v2d/stream', 'c4f')
        self.xy = from_ctypes(self.vlist.vertices,'f8',(N,2))
        self.xy[...] = xy
        self.colors = from_ctypes(self.vlist.colors,'f4',(N, 4))
        self.colors[:,:] = 1.0
    
    def draw(self):
        p = self.params
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0],self.pos[1],0.0)
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_TEXTURE_2D)
        glPointSize(p.dot_size)
        glRotatef(-p.slant,1.0,0.0,0.0)
        glRotatef(p.tilt,0.0,1.0,0.0)        
        glBindTexture(GL_TEXTURE_2D,self.dot.id)
        self.vlist.draw(pyglet.gl.GL_POINTS)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_POINT_SPRITE)
        glPopMatrix()

class Pill(stim(th=0.0,R=50.0,d=3.0,alpha=1.0)):
    """Orientation circle stimulus
    
    :param th: size of stimulus
    :param R: radius
    :param d: width of orientation bar
    """
    def __init__(self,pos,params=Params()):
        self.params = copy_params(self._defaults,params)
        self.pos = pos
        p = self.params
        ld = lambda t,x,y: abs(-sin(t)*x+cos(t)*y)
        w2 = int(ceil(p.R*1.2))
        y,x = mgrid[w2:-w2-1:-1,-w2:w2+1]
        apill = (255*g2f(x/p.R,y/p.R,1.0,40.0)).astype('u1')
        apill = (apill * (1-exp(-0.5*(ld(p.th,x,y)/p.d)**4.0))).astype('u1')
        h,w = apill.shape
        self.image = pyglet.image.ImageData(w,h,'L',apill.tostring(),-w)
        self.image.anchor_x = w//2
        self.image.anchor_y = h//2
    def draw(self):
        glColor4f(1.0,1.0,1.0,self.params.alpha)
        self.image.blit(self.pos[0],self.pos[1])

class OrientCircle(stim(th=0.0,R=50.0,d=3.0,alpha=1.0)):
    """Orientation circle stimulus
    
    :param th: size of stimulus
    :param R: radius
    :param d: width of orientation bar
    """
    frag_source = """
    uniform float th, sigma, gap, alpha;
    float edge = 40.0;

    float lined(float t, float x, float y) {
        return abs(-sin(t)*x+cos(t)*y);
    }

    float g(float x, float s2, float e2) {
        return exp(-0.5*pow(abs(x)/s2,e2));
    }

    void main( void ) {
        float x = gl_TexCoord[0].x - 0.5;
        float y = -(gl_TexCoord[0].y - 0.5);
        float c = exp(-0.5*pow(x*x + y*y,edge/2.0)/pow(sigma,edge));
        c = c*(1.0-g(lined(th,x,y),gap,4.0));
        gl_FragColor = vec4( c,c,c,alpha );

    }
    """
    def __init__(self,pos,params=Params()):
        self.pos = pos
        self.params = copy_params(self._defaults,params)
        p = self.params
        self.shader = Shader(self.frag_source)
        self.program = self.shader.program
        self.uniforms = dict(map(self.shader.uniform,['th','sigma','gap','alpha']))
        glUseProgram(self.program)
        self.w = p.R*2.5
        glUniform1f(self.uniforms['th'],p.th)
        glUniform1f(self.uniforms['sigma'],0.4)
        glUniform1f(self.uniforms['gap'],p.d/self.w)
        glUniform1f(self.uniforms['alpha'],p.alpha)
        glUseProgram(0)
    
    def draw(self):
        p = self.params
        x, y = self.pos
        w2 = self.w/2.0
        glUseProgram(self.program)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0],self.pos[1],0.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0,1.0)
        glVertex2f(-w2,-w2)
        glTexCoord2f(1.0,1.0)
        glVertex2f(w2,-w2)
        glTexCoord2f(1.0,0.0)
        glVertex2f(w2,w2)
        glTexCoord2f(0.0,0.0)
        glVertex2f(-w2,w2)
        glEnd()
        glPopMatrix()
        glUseProgram(0)

class Text(stim(msg='Text',size=24)):
    """Text stimulus
    
    :param msg: string to display
    :params size: font size (default 24)
    """
    def __init__(self,pos,params=Params()):
        self.params = copy_params(self._defaults,params)
        self.pos = pos
        p = self.params
        self.label = text.Label(p.msg,
                          font_name='Times New Roman',
                          font_size=p.size,
                          anchor_x='center',
                          anchor_y='center',
                          x=pos[0], y=pos[1])
    def draw(self):
        glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)
        self.label.draw()

class Dot(stim(c=1.0)):
    """Gaussian dot stimulus
    
    :param c: contrast [0.0,1.0]
    """
    def __init__(self,pos,params=Params()):
        self.params = copy_params(self._defaults,params)
        im = make_dot(0.17,0.0,0.0,2.0,64)
        w = im.width
        h = im.height
        im.anchor_x = w//2
        im.anchor_y = h//2
        self.image = im
        self.pos = pos
    def draw(self):
        c = self.params.c
        glColor4f(c,c,c,1.0)
        self.image.blit(self.pos[0],self.pos[1])


class RDK(stim(width=200.0,n0=500,n1=500,d0=[0.0,1.0],d1=[0.3,-1.0],
               theta=0.0,tilt=0.0,slant=0.0)):
    """Random Dot Kinematogram

    Stimulus consists of two groups of dots. Dots within groups
    move together, motion of each group can be controlled independently.
    
    :param width: width and height in pixels
    :param n0: number of dots in group 1
    :param n1: number of dots in group 2
    :param d0: X,Y speed of group 1 (xy[t+1] = xy[t] + d0)
    :param d1: X,Y speed of group 2 (xy[t+1] = xy[t] + d0)
    :param theta: rotation in screen plane
    :param tilt: tilt angle
    :param slant: slant angle
    """
    def __init__(self,pos,params=Params()):
        self.pos = pos
        self.batch = pyglet.graphics.Batch()
        self.params = copy_params(self._defaults,params)
        p = self.params
        self.clock = pyglet.clock.Clock()
        self.t0 = self.clock.time()
        N = p.n0 + p.n1
        self.dot = make_dot(0.25,0.0,0.0,4.0).get_texture()
        self.vlist = self.batch.add(N, GL_POINTS, None, 'v3d', 'c4f/stream')
        self.xy = from_ctypes(self.vlist.vertices,'f8',(N,3))
        self.xy0 = self.xy[:p.n0,:2]   # Group0 xy coordinates
        self.xy1 = self.xy[p.n0:,:2]   # Group1 xy coordinates
        self.xy0[...] = rand(p.n0,2)*r_[p.width,p.width]
        self.xy1[...] = rand(p.n1,2)*r_[p.width,p.width]
        self.colors = from_ctypes(self.vlist.colors,'f4',(N,4))
        self.colors[...] = 1.0
        self.vlist._vertices_cache.invalidate()
        self.vlist._colors_cache.invalidate()
    
    def draw(self):
        p = self.params
        t = self.clock.time()-self.t0
        self.xy0[...] = (self.xy0 + array(p.d0))%p.width
        self.xy1[...] = (self.xy1 + array(p.d1))%p.width
        self.vlist._vertices_cache.invalidate()
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0]-p.width/2.0,self.pos[1]-p.width/2.0,0.0)
        glRotatef(-p.slant,1.0,0.0,0.0)
        glRotatef(p.tilt,0.0,1.0,0.0)
        glRotatef(p.theta,0.0,0.0,1.0)        
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_TEXTURE_2D)
        glPointSize(10.0)
        glBindTexture(GL_TEXTURE_2D,self.dot.id)
        self.vlist.draw(pyglet.gl.GL_POINTS)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_POINT_SPRITE)
        glPopMatrix()

class RDKMod(stim(width=200.0,n=500,theta=0.0,a=1.0,b=1.0,
                  f=0.0,phi=0.0,noise=0.0,life=500.0,gamma=0.0,
                  tilt=0.0,slant=0.0)):
    """Random Dot Kinematogram with modulated speed

    Dots move together along axis controlled by "theta" parameter.
    Speed of each dot is controlled by

      vx = a + b*cos(2*pi*f*cos(gamma)*x[0] + phi)
      vy = a + b*cos(2*pi*f*sin(gamma)*y[0] + phi)

    The speed is computed from the initial location (x[0],y[0]) 
    of each dot. Then
       
      R = rand-0.5
      x[t+1] = x[t] + vx + vx*noise*R
      y[t+1] = y[t] + vy + vy*noise*R
    
    :param width: width and height in pixels
    :param n: number of dots
    :param theta: motion direction in radians
    :param gamma: angle of modulation relative to the motion direction in radians
    :param a: speed modulation parameter a
    :param b: speed modulation parameter b
    :param f: speed modulation parameter f
    :param phi: speed modulation parameter phi
    :param tilt: tilt angle
    :param slant: slant angle
    """
    def __init__(self,pos,params=Params()):
        self.pos = pos
        self.batch = pyglet.graphics.Batch()
        self.params = copy_params(self._defaults,params)
        p = self.params
        self.clock = pyglet.clock.Clock()
        self.t0 = self.clock.time()
        N = p.n
        self.dot = make_dot(0.25,0.0,0.0,4.0).get_texture()
        self.vlist = self.batch.add(N, GL_POINTS, None, 'v3d', 'c4f/stream')
        self.xy = from_ctypes(self.vlist.vertices,'f8',(N,3))
        self.xy0 = self.xy[:,:2]   # Group0 xy coordinates
        self.xy0[...] = rand(p.n,2)
        #v = p.a + p.b*cos(2*pi*p.f*self.xy0[:,1] + p.phi)
        rx = cos(p.gamma)*self.xy0[:,0] + sin(p.gamma)*self.xy0[:,1]
        v = p.a + p.b*cos(2*pi*p.f*rx + p.phi)
        self.xy0 *= r_[p.width,p.width]
        self.speed = [1.0,0.0]*v.reshape(-1,1)
        self.colors = from_ctypes(self.vlist.colors,'f4',(N,4))
        self.colors[...] = 1.0
        self.vlist._vertices_cache.invalidate()
        self.vlist._colors_cache.invalidate()
        self.draw = self.draw0

    def draw0(self):
        self.life = self.clock.time() + self.params.life/1000.0*rand(self.params.n)
        self.draw = self.draw1
    
    def draw1(self):
        p = self.params
        i = self.clock.time() - self.life > p.life/1000.0
        #m = lambda x: x//1.0 + arcsin(2*(x%1.0)-1.0)/pi+0.5
        #xx = linspace(0,1,100)
        #a = linspace(-1,1,200)
        #hist((m(2*rand(500)-0.75)+m(0.75))/2.0,51,normed=True)
        #plot(xx,sin(2*pi*2.0*xx-0.75*pi/2)+1)
        newxy = rand(i.sum(),2)
        #v = p.a + p.b*cos(2*pi*p.f*newxy[:,1] + p.phi)
        rx = cos(p.gamma)*newxy[:,0] + sin(p.gamma)*newxy[:,1]
        v = p.a + p.b*cos(2*pi*p.f*rx + p.phi)
        self.xy0[i,:] = r_[p.width,p.width]*newxy
        self.speed[i,:] = [1.0,0.0]*v.reshape(-1,1)
        self.life[i,:] = self.clock.time() + self.params.life/1000.0*rand(i.sum())
        vn = p.noise*(rand(p.n)-0.5)
        self.xy0[...] = (self.xy0 + self.speed + self.speed*vn.reshape(-1,1))%p.width
        self.vlist._vertices_cache.invalidate()
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0],self.pos[1],0.0)
        glRotatef(180.0*p.theta/pi,0.0,0.0,1.0)
        glRotatef(-p.slant,1.0,0.0,0.0)
        glRotatef(p.tilt,0.0,1.0,0.0)
        glTranslatef(-p.width/2.0,-p.width/2.0,0.0)
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_TEXTURE_2D)
        glPointSize(10.0)
        glBindTexture(GL_TEXTURE_2D,self.dot.id)
        self.vlist.draw(pyglet.gl.GL_POINTS)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_POINT_SPRITE)
        glPopMatrix()

class Rectangle(stim(width=100.0,height=100.0,angle=0.0,linewidth=1.0,
                     facecolor=(1.0,1.0,1.0,1.0),edgecolor=[])):
    def __init__(self,pos,params=Params()):
        from itertools import cycle
        self.pos = pos
        if hasattr(params,'facecolor') and not iterable(params.facecolor):
            params.facecolor = [params.facecolor]*3
        self.params = copy_params(self._defaults,params)
        p = self.params
        self.clock = pyglet.clock.Clock()
        self.t0 = self.clock.time()
        # make indexed vertex_list
        ind = array([0,1,2,0,2,3])
        xy = r_[0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0].reshape(-1,2) - 0.5
        self.vlist = pyglet.graphics.vertex_list_indexed(4, ind, 'v2d/stream', 'c4f')
        N = 4
        self.xy = from_ctypes(self.vlist.vertices,'f8',(N,2))
        self.xy[...] = xy * r_[p.width,p.height]
        self.colors = from_ctypes(self.vlist.colors,'f4',(N, 4))
        self.colors[:,3] = 1.0
        self.colors[:,:len(p.facecolor)] = p.facecolor
        self.vlist._vertices_cache.invalidate()
        self.vlist._colors_cache.invalidate()
        # outline
        self.edgelist = pyglet.graphics.vertex_list(4, 'v2d/stream', 'c4f')
        self.edgexy = from_ctypes(self.edgelist.vertices,'f8',(N,2))
        self.edgexy[...] = self.xy
        self.edgecolors = from_ctypes(self.edgelist.colors,'f4',(N, 4))
        self.edgecolors[:,3] = 1.0
        self.edgecolors[:,:len(p.edgecolor)] = p.edgecolor
        self.edgelist._vertices_cache.invalidate()
        self.edgelist._colors_cache.invalidate()
        # 
        if not iterable(p.angle): p.angle = [p.angle]
        self.angle = cycle(p.angle)
    
    def draw(self):
        p = self.params
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0],self.pos[1],0.0)
        glRotatef(self.angle.next(),0.0,0.0,1.0)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_MULTISAMPLE_ARB)
        #glEnable(GL_POLYGON_SMOOTH);
        #glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        if p.facecolor: self.vlist.draw(GL_TRIANGLES)
        if p.edgecolor:
            glLineWidth(p.linewidth)
            self.edgelist.draw(GL_LINE_LOOP)
        glEnable(GL_TEXTURE_2D)
        #glDisable(GL_POLYGON_SMOOTH);
        #glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE)
        glDisable(GL_MULTISAMPLE_ARB)
        glPopMatrix()

class Cube(stim(width=100.0,height=100.0,depth=100.0,angles=0.0,linewidth=1.0,
                facecolor=(1.0,1.0,1.0,1.0),edgecolor=[])):
    def __init__(self,pos,params=Params()):
        from itertools import cycle
        self.pos = pos
        if hasattr(params,'facecolor') and not iterable(params.facecolor):
            params.facecolor = [params.facecolor]*3
        self.params = copy_params(self._defaults,params)
        p = self.params
        self.clock = pyglet.clock.Clock()
        self.t0 = self.clock.time()
        # make indexed vertex_list
        ind = array([0,1,2,0,2,3,
                     1,5,6,1,6,2,
                     5,4,7,5,7,6,
                     4,0,3,4,3,7,
                     0,1,5,0,5,4,
                     3,2,6,3,6,7])
        xy = r_[0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,1.0,0.0, 0.0,1.0,0.0,
                0.0,0.0,1.0, 1.0,0.0,1.0, 1.0,1.0,1.0, 0.0,1.0,1.0].reshape(-1,3) - 0.5
        N = xy.shape[0]
        self.vlist = pyglet.graphics.vertex_list_indexed(N, ind, 'v3d/stream', 'c4f')    
        self.xy = from_ctypes(self.vlist.vertices,'f8',(N,3))
        self.xy[...] = xy * r_[p.width,p.height,p.depth]
        self.colors = from_ctypes(self.vlist.colors,'f4',(N, 4))
        self.colors[:,3] = 1.0
        self.colors[:,:len(p.facecolor)] = p.facecolor
        self.vlist._vertices_cache.invalidate()
        self.vlist._colors_cache.invalidate()
        # outline
        indw = array([0,1,1,2,2,3,3,0,
                      1,5,5,6,6,2,2,1,
                      5,4,4,7,7,6,6,5,
                      4,0,0,3,3,7,7,4,
                      0,1,1,5,5,4,4,0,
                      3,2,2,6,6,7,7,3])
        self.edgelist = pyglet.graphics.vertex_list_indexed(N, indw, 'v3d/stream', 'c4f')
        self.edgexy = from_ctypes(self.edgelist.vertices,'f8',(N,3))
        self.edgexy[...] = self.xy
        self.edgecolors = from_ctypes(self.edgelist.colors,'f4',(N, 4))
        self.edgecolors[:,3] = 1.0
        self.edgecolors[:,:len(p.edgecolor)] = p.edgecolor
        self.edgelist._vertices_cache.invalidate()
        self.edgelist._colors_cache.invalidate()
        # 
        if not iterable(p.angle): p.angle = [p.angle]
        self.angle = cycle(p.angle)
    
    def draw(self):
        p = self.params
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pos[0],self.pos[1],0.0)
        glRotatef(self.angle.next(),0.0,1.0,1.0)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_MULTISAMPLE_ARB)
        #glEnable(GL_POLYGON_SMOOTH);
        #glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        if p.facecolor: self.vlist.draw(GL_TRIANGLES)
        if p.edgecolor:
            glLineWidth(p.linewidth)
            self.edgelist.draw(GL_LINES)
        glEnable(GL_TEXTURE_2D)
        #glDisable(GL_POLYGON_SMOOTH);
        #glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE)
        glDisable(GL_MULTISAMPLE_ARB)
        glPopMatrix()
