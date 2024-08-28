
from __future__ import division

import os
import sys
sys.path[:0] = ['packages.zip']
import traceback

import time
from ctypes import addressof
from numpy import *
import numpy as np
from numpy.random import rand, randn

def from_ctypes(buf, dtype, shp):
    class Dummy(object): pass 
    d = Dummy() 
    d.__array_interface__ = {
         'data' : (addressof(buf), False), 
         'typestr' : np.dtype(dtype).str,
         'descr' : np.dtype(dtype).descr,
         'shape' : shp,
         'strides' : None,
         'version' : 3
    }
    return np.asarray(d).view(dtype=dtype)

g = lambda x_,s_=1.0,p_=2.0: exp(-0.5*pow(abs(x_)**2,p_/2.0)/pow(s_,p_))
g2f = lambda x_,y_,s_,p_=2.0: exp(-0.5*(abs(x_)**2.0 + abs(y_)**2.0)**(p_/2.0)/(s_)**p_)
best_sigma = lambda w_,y_,p_: pow(-0.5*pow(abs(w_)**2,p_/2.0)/log(y_),1.0/p_)
xg = lambda y_,s_,p_: pow(-2.0*log(y_)*pow(s_,p_),1.0/p_)

def mat(*x):
    return np.array(x).reshape(3,3)
def rmatx(a):
    ca = np.cos(a); sa = np.sin(a)
    return mat(1,0,0,0,ca,sa,0,-sa,ca)
def rmatz(a):
    ca = np.cos(a); sa = np.sin(a)
    return mat(ca,sa,0,-sa,ca,0,0,0,1)
def slanttilt(X,sl,tl,pt):
    R = np.dot(rmatx(sl),rmatz(tl))
    return np.dot(X-pt,R)+pt

# define rectangles in normalized coordinates of the image frame
rc3 = np.array([[0.0,0.0,0.0],
             [1.0,0.0,0.0],
             [1.0,1.0,0.0],
             [0.0,1.0,0.0]])
rc3c = rc3 - np.r_[0.5,0.5,0.0]; # now we have a centered unit rectangle 

def makecrect(cx,cy,w,h):
    return np.r_[w,h,0]*rc3c + np.r_[cx,cy,0]
def makerect(x0,y0,w,h):
    return np.r_[w,h,0]*rc3 + np.r_[x0,y0,0]


from pyglet.gl import *
from pyglet.window import key
from pyglet import clock, clock, font, graphics, window, text

rads = lambda x: pi*x/180.0
degs = lambda x: 180.0*x/pi

_ttoc = pyglet.clock.Clock()
def tic():
    global _ttoc
    _ttoc = pyglet.clock.Clock()
def toc():
    global _ttoc
    print _ttoc.time()

def ppd(d=67.0,patch_px=1024,patch=38.5):
    from math import pi
    viewingDistance = d           # distance
    patchPixels = patch_px        # the size of the patch in pixels
    patchCentimeters = patch      # the measurement of the same patch
    pixelsPerCentimeter = patchPixels/float(patchCentimeters)
    return viewingDistance * (pi/180) * pixelsPerCentimeter

def make_dot(sigma,fs,phi,edge,res=128):
    [y,x] = mgrid[-1:1:1j*res,-1:1:1j*res]
    xx = cos(phi)*x - sin(phi)*y
    yy = sin(phi)*x + cos(phi)*y
    im = 127*(cos(2*pi*fs*xx)+1)
    im = tile(im.reshape(res,res,1),(1,1,4))
    im[:,:,3] = 255.0*exp(-0.5*pow(x**2 + y**2,edge/2.0)/pow(sigma,edge))
    return pyglet.image.ImageData(res, res, 'RGBA', im.astype('u1').tostring(), -res*4)

def make_lattice(apw=200.0,dx=20.0,r=1.5,gamma=pi/2,theta=0.0):
    x = arange(0,apw,dx)
    x = r_[-x[-1:0:-1],x]
    y = arange(0,apw,dx*r*sin(gamma))
    y = r_[-y[-1:0:-1],y]
    X,Y = meshgrid(x,y)
    X[::2,:] += dx*r*cos(gamma)
    # rotate
    xy = empty((X.size,2),'f8')
    sinT = sin(theta)
    cosT = cos(theta)
    xy[:,0] = X.ravel()*cosT-Y.ravel()*sinT
    xy[:,1] = X.ravel()*sinT+Y.ravel()*cosT
    xy = xy[sqrt((xy**2).sum(1)) < apw,:]
    return xy

def make_mask(w,h,s,p):
    imap = zeros((h,w,4),'u1')
    X,Y = meshgrid(arange(0.0,w),arange(0.0,h))
    w2 = w/2.0
    h2 = h/2.0
    W = float(w)
    X = X-w2
    Y = Y-h2
    g2 = lambda s_,p_=2.0: exp(-0.5*(abs(X/W)**2.0 + abs(Y/W)**2.0)**(p_/2.0)/(s_/W)**p_)
    imap[:,:,3] = 255-255*g2(s,p)
    ap = pyglet.image.ImageData(w,h,'RGBA',imap.tostring(),-w*4)
    ap.anchor_x = int(w2)
    ap.anchor_y = int(h2)
    ap = pyglet.sprite.Sprite(ap)
    return ap

_vp_batches = []
def make_indarray(N):
    global _vp_batches
    inds = (array([0,1,2,0,2,3]) + 4*arange(N).reshape(-1,1)).reshape(-1)
    #return pyglet.graphics.vertex_list_indexed(4*N, inds, 'v2d', 'c4f/stream', 't2f'), inds
    batch = pyglet.graphics.Batch()
    _vp_batches += [batch]
    return batch, batch.add_indexed(4*N, GL_TRIANGLES, None, inds, 'v3d', 'c4f/stream', 't2f'), inds

def boxlut(c, Lbg=49.3/2.0, Lmin=0.0, Lmax=49.3, gamma=2.5, BTRR = 63.2):
    Lc = (1+c)*Lbg
    U = 255.0*((Lc-Lmin)/(Lmax-Lmin))**(1/gamma)
    b1 = (BTRR+1)/BTRR;
    b = np.fmin(np.floor(U*b1), 255)
    return (around((U-b/b1)*(BTRR+1))/255.0, 0, around(b)/255.0)

from ctypes import byref, c_char_p, c_char, cast, POINTER, create_string_buffer
class Shader:
    def __init__(self,frag_source,vert_source=''):
        self.program = glCreateProgramObjectARB()
        if frag_source: self.create_shader(frag_source,GL_FRAGMENT_SHADER)
        if vert_source: self.create_shader(vert_source,GL_VERTEX_SHADER)
        glLinkProgram(self.program)
        self.check_errors(self.program, GL_OBJECT_LINK_STATUS_ARB, errormsg="failed to link:\n%s")
        glValidateProgram(self.program)
        self.check_errors(self.program, GL_VALIDATE_STATUS, errormsg="failed to validate:\n%s")
    def create_shader(self,src,stype):
        source_ptr = cast(c_char_p(src), POINTER(c_char))
        sh = glCreateShaderObjectARB(stype)
        glShaderSource(sh, 1, byref(source_ptr), None)
        glCompileShader(sh)
        self.check_errors(sh, GL_OBJECT_COMPILE_STATUS_ARB, errormsg="failed to compile:\n%s")
        glAttachObjectARB(self.program, sh)
    def shader_log(self,obj):
        length = GLint(0)
        glGetObjectParameterivARB(obj, GL_OBJECT_INFO_LOG_LENGTH_ARB, byref(length))
        log = create_string_buffer(length.value)
        glGetInfoLogARB(obj, length.value, None, log)
        return log.value
    def check_errors(self,obj,type,errormsg):
        status = GLint(0)
        glGetObjectParameterivARB(obj, type, byref(status))
        if status.value == 0:
            error = self.shader_log(obj)
            raise Exception(errormsg % error)
    def uniform(self,name):
        value = glGetUniformLocation(self.program,name)
        return (name, value)

class Params:
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
    def update(self,**kwargs):
        self.__dict__.update(kwargs)
    def __str__(self):
        return '\n'.join( '%s = %s'%x for x in self.__dict__.items() )
    def __repr__(self):
        return 'Params(%s)'%(', '.join( '%s=%s'%x for x in self.__dict__.items() ),)
    def default(self):
        return self.__dict__

def copy_params(p,params=None,**kwargs):
    out = Params()
    for k,v in p.__dict__.items():
        if k[0] != '_': setattr(out,k,v)
    if params is not None:
        if isinstance(params,Params):
            out.update(**params.__dict__)
        else:
            self.params.update(params)
    out.update(**kwargs)
    return out

from ctypes import *
class FBO(object):
    """Basic helper for using OpenGL's Frame Buffer Object (FBO)"""
    
    @staticmethod
    def supported():
        """A static method that tells if FBOs are supported.
        If not sure, call this before creating an FBO instance."""
        
        # Check if the board / driver supports FBO
        if not gl_info.have_extension("GL_EXT_framebuffer_object"):
            return False
        if not gl_info.have_extension("GL_ARB_draw_buffers"):
            return False
        
        return True

    def __init__(self, width=100, height=100):
        """Creates a Frame Buffer Object (FBO)"""
    
        # Must be supported...
        assert (FBO.supported())
        
        self.width = width
        self.height = height
 
        # Setting it up
        self.framebuffer = c_uint(0)
        self.depthbuffer = c_uint(0)
        self.img = c_uint(0)
        
        glGenFramebuffersEXT(1, byref(self.framebuffer))
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.framebuffer)
        
        # Adding a Depth Buffer
        glGenRenderbuffersEXT(1, byref(self.depthbuffer))
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, self.depthbuffer)
        glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, self.width, self.height)
        glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, 
                                     GL_RENDERBUFFER_EXT, self.depthbuffer)
    
        # Adding a Texture To Render To
        glGenTextures(1, byref(self.img))
        glBindTexture(GL_TEXTURE_2D, self.img)
    
        # Black magic (only works with these two lines)
        # (nearest works, as well as linear)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)

        # Add the texture ot the frame buffer as a color buffer
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 
                     0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, 
                     GL_TEXTURE_2D, self.img, 0)
    
        # Check if it worked so far
        status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)
        assert(status == GL_FRAMEBUFFER_COMPLETE_EXT)

    def attach(self):
        """Call this before rendering to the FBO."""

        # First we bind the FBO so we can render to it
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.framebuffer)
        
        # Save the view port and set it to the size of the texture
        glPushAttrib(GL_VIEWPORT_BIT)
        glViewport(0,0,self.width,self.height)

    def detach(self):
        """Call this after rendering to the FBO so that rendering now
        goes to the regular frame buffer."""

        # Restore old view port and set rendering back to default frame buffer
        glPopAttrib()    
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0)

    def getTexture(self):
        """Returns a pyglet image with the contents of the FBO."""
        self.data = (c_ubyte * (self.width*self.height*4))()
    
        glGetTexImage(GL_TEXTURE_2D, # target, 
                          0, # level, 
                          GL_RGBA, # format, 
                          GL_UNSIGNED_BYTE , # type, 
                          self.data) # GLvoid * img
                      
        return pyglet.image.ImageData (self.width, self.height, 'RGBA', self.data)
        
    def __del__(self):
        """Deallocates memory. Call this before discarding FBO."""
        glDeleteFramebuffersEXT(1, byref(self.framebuffer))
        glDeleteRenderbuffersEXT(1, byref(self.depthbuffer))
        glDeleteTextures(1,byref(self.img))

def stim_default(self):
    return {'type':'Stimulus','name':self.__class__.__name__,'pos':self.pos,'params':self.params}

def stim(**kwargs):
    return type('Stimulus',(),{'_defaults':Params(**kwargs),'default':stim_default})

def stim_factory(**kwargs):
    class Stimulus(object):
        _defaults = Params(**kwargs)
        def __init__(self,params=None,**kw):
            self.params = copy_params(self._defaults)
            if params is not None:
                if isinstance(params,Params):
                    self.params.__dict__.update(params.__dict__)
                else:
                    self.params.update(params)
            self.params.__dict__.update(kw)
    return Stimulus

from Queue import Empty
_exp_windows = []
class ExpWindow(window.Window):
    def __init__(self,fullscreen=True,**kwargs):
        global _exp_windows
        super(ExpWindow,self).__init__(fullscreen=fullscreen,vsync=True,**kwargs)
        _exp_windows.append(self)
        self.run = True
        glEnable(GL_POINT_SMOOTH)
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glClearDepth(1.0)               # Depth buffer setup
        #glEnable(GL_DEPTH_TEST)         # Enables depth testing
        #glDepthFunc(GL_LEQUAL)          # The type of depth test to do
        glPointSize(10.0)
        glClearColor(0.0,0.0,0.0,1.0)
        self.trials = []
        self.lastevname = ''
        self.ctrial = None
        self.logger = None
        self.remote = None
        self.clock = pyglet.clock.Clock()
        self.on_draw = self.on_draw_idle
        self.idle_msg = text.Label('mPsy ready! No trials in the queue.',
                            font_name='Times New Roman',
                            font_size=24,
                            anchor_x='center',
                            anchor_y='center',
                            x=self.width//2, y=self.height//2)
        self.dispatch_events()
        self.idle_msg.draw()
        self.flip()

        #self.set_exclusive_mouse(True)
    def __del__(self):
        if self.remote:
            self.remote.stop()
            time.sleep(0.5)
    def exit(self):
        if self.remote:
            self.remote.stop()
            self.remote = None
        time.sleep(0.1)
        pyglet.app.exit()

    def update_state(self,msg,flip=True):
        self.idle_msg.text = msg
        self.dispatch_events()
        self.clear()
        self.idle_msg.draw()
        # need to flip in order to get the new message
        # override default if desired
        if flip: self.flip()
    
    def update(self,dt):
        pass
    def _update_remote(self,dt):
        qcmd, qret = self.remote.qcmd, self.remote.qret
        while not qcmd.empty():
            try:
                cmd, args = qcmd.get(False)
                self.event('REMOTE',(cmd,args))
                if cmd == 'set_trials':
                    trials = self.remote.trials_from_json(args)
                    self.set_trials(trials)
                    qret.put('OK')
                elif cmd == 'add_trials':
                    trials = self.remote.trials_from_json(args)
                    self.add_trials(trials)
                    qret.put('OK')
                elif cmd == 'alive':
                    qret.put('OK')
                elif cmd == 'set_background':
                    self.set_background(args)
                    qret.put('OK')
                elif cmd == 'update_state':
                    self.update_state(args)
                    qret.put('OK')
                elif cmd == 'info':
                    qret.put('[%d,%d,%d]'%(self.width,self.height,self._hwnd))
                elif cmd == 'events':
                    ev = self.remote.eventlog
                    self.remote.eventlog = []
                    qret.put(self.remote.translate_events(ev))
                elif cmd == 'quit':
                    qret.put('OK')
                    time.sleep(0.1)
                    self.exit()
                else:
                    qret.put('ERROR: unknown command: "%s"'%cmd)
            except Empty:
                pass
            except Exception, e:
                self.event('EXCEPTION','_update_remote')
                tp,val,tb = sys.exc_info()
                exc_str = traceback.format_tb(tb)
                qret.put('ERROR: EXCEPTION in "_update_remote":\n'+'\n'.join(exc_str))
    
    def on_resize(self,width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        #glOrtho(0,width,0,height,-1000.0,1000.0)
        zp = 500.0
        w,h = width,height
        #gluPerspective( 180.0*(2*arctan(h/(2*zp)))/pi, w/h, zp/2.0, 3*zp/2.0)
        gluPerspective( 180.0*(2*arctan(h/(2*zp)))/pi, w/h, 200.0, 800.0)
        cx, cy = self.width/2.0, self.height/2.0
        glTranslatef(-cx,-cy,-zp)
        glMatrixMode(GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED
    
    def on_draw_idle(self):
        self.clear()
        if self.trials:
            self.ctrial = self.trials.pop(0)
            self.event('TRIAL')
            self.set_exclusive_mouse(not self.ctrial.mouse)
            self.T = time.time() + self.ctrial.T/1000.0
            self.on_draw = self.on_draw_trial
            self.keys = self.ctrial.keys
            self.ctrial.draw(self)
        else:
            glEnable(GL_BLEND)
            glEnable(GL_TEXTURE_2D)
            self.idle_msg.draw()
            self.event('EMPTY')
    
    def on_draw_trial(self):
        self.ctrial.draw(self)
        if self.ctrial.T > 0 and time.time() >= self.T:
            self.on_draw = self.on_draw_idle
        elif not self.run:
            self.on_draw = self.on_draw_idle
            self.run = True
    
    def on_key_press(self,symbol,modifiers):
        if symbol == key.ESCAPE:
            self.exit()
        self.event('KEY',[key._key_names[symbol],symbol])
        if self.ctrial and symbol in self.ctrial.keys:
            self.on_draw = self.on_draw_idle
    
    def on_mouse_press(self,x,y,button,modifiers):
        self.event('MOUSE',[x,y,button])
        if self.ctrial and self.ctrial.mouse:
            self.on_draw = self.on_draw_idle
    
    def event(self,name,data=[]):
        ret = None
        if self.logger:
            if name == 'EMPTY' and name == self.lastevname:
                # do no repeat the EMPTY event
                self.update_state('mPsy ready! No trials in the queue.',False)
                return None
            
            try:
                ret = self.logger(name,self.clock.time(),self.ctrial,data)
            except Exception, e:
                tp,val,tb = sys.exc_info()
                print '\n### mPsy ERROR ##########'
                print val
                print traceback.print_tb(tb)
                print '##############################\n'
                ret = 'next'
            if ret == 'next':
                # skip stimulus
                self.run = False
        self.lastevname = name
        return ret
    
    def set_background(self,color):
        if not iterable(color):
            color = (color,color,color,1.0)
        elif len(color) == 3:
            color = tuple(color)+(1.0,)
        glClearColor(*color)
    
    def set_trials(self,trials):
        self.idle_msg.text = ''
        self.trials = trials
        self.on_draw = self.on_draw_idle
    def add_trials(self,trials):
        self.idle_msg.text = ''
        self.trials.extend(trials)
    def set_logger(self,logger):
        self.logger = logger
    def set_remote(self,remote):
        self.remote = remote
        self.update = self._update_remote

class Trial:
    def __init__(self,name,stimuli,T,keys=[],mouse=False):
        self.stimuli = stimuli
        self.T = T
        self.name = name
        self.keys = keys
        self.mouse = mouse
    def draw(self,win):
        win.clear()    
        for stim in self.stimuli:
            stim.draw()
    def __str__(self):
        return self.name
    def default(self):
        return {'type':'Trial','name':self.name,'keys':self.keys,'mouse':self.mouse,'stimuli':self.stimuli}

exp_time = 0.0
def update(dt):
    global exp_time
    for x in _exp_windows:
        x.update(dt)

def run():
    clock.schedule_interval(update,1/60.0)
    pyglet.app.run()

def mainloop(q,r):
    from Queue import Empty
    
    run = True
    p = None
    while run and p is None:
        func, args = q.get()
        if isinstance(func,str) and func == 'QUIT': run = False
        if func != Presentation:
            r.put('VP2 error: mainloop: Presentation not initialized!')
        else:
            p = Presentation(*args)
            r.put(p)
        q.task_done()
        sleep(0.1)
    
    while run:
        try:
            func, args = q.get(False)
            if isinstance(func,str) and func == 'QUIT':
                run = False
            elif func == Presentation:
                p._current_draw = p.draw_ready
                r.put(p)
            else:
                r.put(func(*args))
        except Empty:
            pass
        else:
            q.task_done()
        run = p.dispatch_events()
