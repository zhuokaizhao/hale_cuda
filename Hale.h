/*
  hale: support for minimalist scientific visualization
  Copyright (C) 2014, 2015  University of Chicago

  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
  arising from the use of this software. Permission is granted to anyone to
  use this software for any purpose, including commercial applications, and
  to alter it and redistribute it freely, subject to the following
  restrictions:

  1. The origin of this software must not be misrepresented; you must not
  claim that you wrote the original software. If you use this software in a
  product, an acknowledgment in the product documentation would be
  appreciated but is not required.

  2. Altered source versions must be plainly marked as such, and must not be
  misrepresented as being the original software.

  3. This notice may not be removed or altered from any source distribution.
*/

#ifndef HALE_INCLUDED
#define HALE_INCLUDED

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <map>
#include <list>
#include <vector>

/* This will include all the Teem headers at once */
#include <teem/meet.h>

/*
** We don't #define GLM_FORCE_RADIANS here because who knows what Hale users
** will expect or need. Hale's own source does use GLM_FORCE_RADIANS but
** that's in privateHale.h
*/
#include <glm/glm.hpp>

/*
** We want to restrict ourself to Core OpenGL, but have found that on at least
** one Linux System, using "#define GLFW_INCLUDE_GLCOREARB" caused things like
** glGetError and glViewport to not be defined.  Conversely, on that same
** linux machine, functions like glCreateShader and glShaderSource were not
** defined unless there was '#define GL_GLEXT_PROTOTYPES", even though
** <https://www.opengl.org/registry/ABI/> suggests that GL_GLEXT_PROTOTYPES is
** something one queries with #ifdef rather than #define'ing.  HEY: Some
** expertise here would be nice.
*/
#if defined(__APPLE_CC__)
#  define GLFW_INCLUDE_GLCOREARB
#else
#  define GL_GLEXT_PROTOTYPES
#endif
/* NB: on at least one Linux system that was missing GL/glcorearb.h,
   GLK followed advice from here:
   http://oglplus.org/oglplus/html/oglplus_getting_it_going.html and
   manually put glcorearb.h in /usr/include/GL, however the info here:
   https://www.opengl.org/registry/api/readme.pdf suggests that all
   those headers should be locally re-generated by a script */
#include <GLFW/glfw3.h>

namespace Hale {

typedef void (*ViewerRefresher)(void*);

/*
** enums.cpp: Various C enums are used to representing things with
** integers, and the airEnum provides mappings between strings and the
** corresponding integers
*/

/*
** viewerMode* enum
**
** The GUI modes that the viewer can be in. In Fov and Depth (distance from
** near to far adjusted), the look-from and look-at point are both fixed. The
** eye moves around a fixed look-at point in the Rotate and Vertigo
** modes. The eye and look-at points move together in the Translate modes.
*/
enum {
  viewerModeUnknown,        /*  0 */
  viewerModeNone,           /*  1: buttons released => no camera
                                interaction */
  viewerModeFov,            /*  2: standard "zoom" */
  viewerModeDepthScale,     /*  3: scale distance between near and far
                                clipping planes */
  viewerModeTranslateN,     /*  4: translate along view direction */
  viewerModeRotateUV,       /*  5: usual rotate (around look-at point) */
  viewerModeRotateU,        /*  6: rotate around horizontal axis */
  viewerModeRotateV,        /*  7: rotate around vertical axis */
  viewerModeRotateN,        /*  8: in-plane rotate (around at point) */
  viewerModeVertigo,        /*  9: fix at, move from, adjust fov: the effect
                                is direct control on amount of perspective
                                (aka dolly zoom, c.f. Hitchcock's Vertigo) */
  viewerModeTranslateUV,    /* 10: usual translate */
  viewerModeTranslateU,     /* 11: translate only horizontal */
  viewerModeTranslateV,     /* 12: translate only vertical */
  viewerModeZoom,           /* 13: scale from-at distance by moving from
                               (while fixing at), and also scale the clipping
                               plane distances.  The effect is that the world
                               is being scaled relative to camera */
  viewerModeSlider,         /* 14: not really a viewer mode at all, but using
                               interactions to control a "slider" on bottom
                               edge of window */
  viewerModeLast
};
extern airEnum *viewerMode;

/*
** Though it is somewhat limiting, for now we tie these to the
** limnPolyDataInfo (even copying the order!), and the contents of
** limnPolyData. Note however that the limnPolyData->xyzw position
** information is always set, and is thus not included in the
** limnPolyDataInfo enum, but obviously needs to be listed as a vertex
** attribute here
*/
enum {
  vertAttrIdxUnknown = -1, /* -1: (0 is a valid index) */
  vertAttrIdxXYZW,         /*  0: XYZW position */
  vertAttrIdxRGBA,         /*  1: RGBA color */
  vertAttrIdxNorm,         /*  2: 3-vector normal */
  vertAttrIdxTex2,         /*  3: (s,t) texture coords */
  vertAttrIdxTang,         /*  4: unit-length surface tangent 3-vectors */
  vertAttrIdxLast          /*  5 */
};
#define HALE_VERT_ATTR_IDX_NUM 5

enum {
  finishingStatusUnknown,   /* 0 */
  finishingStatusNot,       /* 1: we're still running */
  finishingStatusOkay,      /* 2: we're quitting gracefully */
  finishingStatusError,     /* 3: we're exiting with error */
  finishingStatusLast
};
extern airEnum *finishingStatus;

/*
** GLSL programs that are "pre-programmed"; the source for them is internal
** to Hale
*/
typedef enum {
  preprogramUnknown,            /* 0 */
  preprogramAmbDiff,            /* 1 */
  preprogramAmbDiffSolid,       /* 2 */
  preprogramAmbDiff2Side,       /* 3 */
  preprogramAmbDiff2SideSolid,  /* 4 */
  preprogramLast
} preprogram;

/* globals.cpp */
extern bool finishing;
extern int debugging;

/* utils.cpp */
extern void init();
extern void done();
extern GLuint limnToGLPrim(int type);
extern void glErrorCheck(std::string whence, std::string context);
typedef struct {
  /* copy of the same enum value used for indexing into glEnumDesc */
  GLenum enumVal;
  /* string of the GLenum value, e.g. "GL_FLOAT", "GL_FLOAT_MAT4" */
  std::string enumStr;
  /* string for corresponding glsl type, e.g. "float", "mat4" */
  std::string glslStr;
} glEnumItem;
/* gadget to map GLenum values to something readable */
extern std::map<GLenum,glEnumItem> glEnumDesc;

extern GLuint loadTextureImage(const Nrrd *nimg);
extern GLuint loadTextureImage(const unsigned char *data, int width, int height, int pixsize);

/* Camera.cpp: like Teem's limnCamera but simpler: the image plane is
   always considered to be containing look-at point, there is no
   control of right-vs-left handed coordinates (it is always
   right-handed: U increases to the right, V increases upward, and N
   points towards camera), and clipNear and clipFar are always
   relative to look-at point. */
class Camera {
 public:
  explicit Camera(glm::vec3 from = glm::vec3(3.0f,4.0f,5.0f),
                  glm::vec3 at = glm::vec3(0.0f,0.0f,0.0f),
                  glm::vec3 up = glm::vec3(0.0f,0.0f,1.0f),
                  double fov = 15,
                  double aspect = 1.3333333,
                  double clipNear = -2,
                  double clipFar = 2,
                  bool orthographic = false);

  /* set/get verbosity level */
  void verbose(int);
  int verbose();

  /* set everything at once, as if at initialization */
  void init(glm::vec3 from, glm::vec3 at, glm::vec3 up,
            double fov, double aspect,
            double clipNear, double clipFar,
            bool orthographic);

  /* set/get world-space look-from, look-at, and pseudo-up */
  void from(glm::vec3); glm::vec3 from();
  void at(glm::vec3);   glm::vec3 at();
  void up(glm::vec3);   glm::vec3 up();

  /* make up orthogonal to at-from */
  void reup();

  /* setters, getters */
  void fov(double);        double fov();
  void aspect(double);     double aspect();
  void clipNear(double);   double clipNear();
  void clipFar(double);    double clipFar();
  void orthographic(bool); bool orthographic();

  /* the (world-to-)view and projection transforms
     determined by the above parameters */
  glm::mat4 view();
  glm::mat4 viewInv();
  glm::mat4 project();
  const float *viewPtr();
  const float *projectPtr();

  /* basis vectors of view space */
  glm::vec3 U();
  glm::vec3 V();
  glm::vec3 N();

  /* generate string for command-line options */
  std::string hest();

 protected:
  int _verbose;

  /* essential camera parameters */
  glm::vec3 _from, _at, _up;
  double _fov, _aspect, _clipNear, _clipFar;
  bool _orthographic;

  /* derived parameters */
  glm::vec3 _uu, _vv, _nn; // view-space basis
  glm::mat4 _view, _viewInv, _project;

  void updateView();
  void updateProject();
};

class Scene;  // (forward declaration)

/* Viewer.cpp: Viewer contains and manages a GLFW window, including the
   camera that defines the view within the viewer.  We intercept all
   the events in order to handle how the camera is updated */
class Viewer {
 public:
  explicit Viewer(int width,  int height, const char *label, Scene *scene);
  ~Viewer();

  /* the camera we update with user interactions */
  Camera camera;

  /* set/get verbosity level */
  void verbose(int);
  int verbose();

  /* set window title */
  void title();

  /* get current interaction mode */
  int mode() const;

  /* get width and height of window in screen-space, which is not always
     the same as dimensions of frame buffer (on high-DPI displays, the
     frame buffer size is larger than the nominal size of the window).
     There are no methods for setting width and height because that's
     handled by responding to GLFW resize events */
  int width();
  int height();

  /* set/get whether to fix the "up" vector during camera movements */
  void upFix(bool);
  bool upFix();

  /* set/get refresh callback, and its data */
  void refreshCB(ViewerRefresher cb);
  ViewerRefresher refreshCB();
  void refreshData(void *data);
  void *refreshData();

  /* swap render buffers in window */
  void bufferSwap();

  /* makes context of this GLFW window current */
  void current();

  /* save RGBA of current window to file */
  void snap(const char *fname); // to this filename
  void snap();                  // to some new file

  /* set/get scene */
  const Scene *scene();
  void scene(Scene *scn);

  /* set/get view-space light position */
  void lightDir(glm::vec3 dir);
  glm::vec3 lightDir(void) const;

  /* print usage info */
  void helpPrint(FILE *) const;

  /* relating to using bottom edge as slider */
  void slider(double *slvalue, double min, double max);
  bool slidable() const;
  bool sliding() const;
  void sliding(bool);

  /* toggling an on/off variable via space bar */
  void toggle(int *tvalue);

  /* extra camera information */
  std::string origRowCol();

 /* we can return a const Scene* via scene(), but then the caller can't
    draw() it; this draw() just calls the scene's draw() */
  void draw(void);

  bool isMouseReleased();
  bool isMasked();

 protected:
  glm::vec3 _lightDir;
  bool _button[2];     // true iff button (left:0, right:1) is down
  std::string _label;
  Scene *_scene;
  int _verbose;
  bool _upFix;
  int _mode;           // from Hale::viewerMode* enum
  ViewerRefresher _refreshCB;
  void * _refreshData;

  int _pixDensity,
    _widthScreen, _heightScreen,
    _widthBuffer, _heightBuffer;
  // space to get current windows pixel values
  unsigned char *_buffRGBA[2];
  Nrrd *_nbuffRGBA[2];
  void _buffAlloc(void); // manages previous _buff* and _nbuff*
  double _lastX, _lastY; // last clicked position, in screen space
  bool _slidable, // can toggle to using right-click on bottom edge as slider
    _sliding;     // is now being used as slider
  void _slrevalue(const char *me, double xx);
  double *_slvalue, // value to modify via slider
    _slmin, _slmax;  // range of possible slider values
  int *_tvalue; // value to toggle via space bar
  int _stateMasked;

  GLFWwindow *_window; // the window we manage
  static void cursorPosCB(GLFWwindow *gwin, double xx, double yy);
  static void framebufferSizeCB(GLFWwindow *gwin, int newWidth, int newHeight);
  static void keyCB(GLFWwindow *gwin, int key, int scancode, int action, int mods);
  static void windowCloseCB(GLFWwindow *gwin);
  static void windowRefreshCB(GLFWwindow *gwin);
  static void mouseButtonCB(GLFWwindow *gwin, int button, int action, int mods);

  void shapeUpdate();
};

/*
** Program.cpp: a GLSL shader program contains shader objects for vertex and
** fragment shaders (can easily add a geometry shader when needed)
*/
class Program {
 public:
  explicit Program(preprogram prog);
  explicit Program(const char *vertFname, const char *fragFname);
  ~Program();
  void compile();
  void bindAttribute(GLuint idx, const GLchar *name);
  void link();
  GLuint progId() const;
  void use() const;

  // will add more of these as needed
  void uniform(std::string, float, bool sticky=false) const;
  void uniform(std::string, int, bool sticky=false) const;
  void uniform(std::string, glm::vec3, bool sticky=false) const;
  void uniform(std::string, glm::vec4, bool sticky=false) const;
  void uniform(std::string, glm::mat4, bool sticky=false) const;
  // these are the basis of uniform()'s implementation, and they should
  // perhaps be private, but this way they're accessible to experts
  std::map<std::string,GLint> uniformLocation;
  std::map<std::string,glEnumItem> uniformType;
 protected:
  GLint _vertId, _fragId, _progId;
  GLchar *_vertCode, *_fragCode;

};
/* Extra functions not in Program: ways to communicate (really,
   broadcast, or shout) uniforms to whatever is current program */
extern void uniform(std::string, float, bool sticky=false);
extern void uniform(std::string, glm::vec3, bool sticky=false);
extern void uniform(std::string, glm::vec4, bool sticky=false);
extern void uniform(std::string, glm::mat4, bool sticky=false);
/* The "sticky uniforms"; Program->use() calls stickyUniform() to re-set
   all the uniforms that were intended to be used for all programs */
extern std::map<std::string, float> stickyUniformFloat;
extern std::map<std::string, glm::vec3> stickyUniformVec3;
extern std::map<std::string, glm::vec4> stickyUniformVec4;
extern std::map<std::string, glm::mat4> stickyUniformMat4;
extern void stickyUniform(void);
/* way to access one of the "pre-programs"; will compile as needed */
extern const Program *ProgramLib(preprogram pp); 
extern const Program *ProgramLib(const char *vertFname, const char *fragFname, const char *nameTexture,
                const char *nameXYZW, const char *nameRGBA, const char *nameNorm, const char *nameTex2);

class Polydata {
 public:
  explicit Polydata(const limnPolyData *poly,  // don't own
                    const Program *prog, std::string name="");
  explicit Polydata(limnPolyData *poly, bool own, // may or may not own
                    const Program *prog, std::string name="");
  ~Polydata();
  /* if you want to get the underlying limn representation */
  const limnPolyData *lpld() const { return _lpld ? _lpld : _lpldOwn; }
  void rebuffer();            // glBuffer(Sub)Data calls

  /* set/get constant color, *if* there is no per-vertex color */
  void colorSolid(float rr, float gg, float bb);
  void colorSolid(glm::vec3 rgb);
  void colorSolid(glm::vec4 rgba);
  glm::vec4 colorSolid() const;
  void setTexture(char *varName, Nrrd *nimg);
  void setTexture(char *varName, const unsigned char *data, int width, int height, int pixsize);
  void replaceLastTexture(unsigned char *data, unsigned int width, unsigned int height, unsigned int pixsize);
  void bindTexture() const;
  /* set/get model transformation */
  void model(glm::mat4 mat);
  glm::mat4 model() const;

  /* set/get program to use when drawing */
  void program(const Program *);
  const Program *program() const;

  void bounds(glm::vec3 &min, glm::vec3 &max) const;
  void draw() const;

  /* set/get object "name" */
  void name(std::string nm);
  std::string name() const;

 protected:
  std::string _name;
  GLuint _vao;                // GL vertex array object
  glm::vec4 _colorSolid;      // constant color
  glm::mat4 _model;           // object to world transform
  void _init(std::string);   // main constructor body
  void _buffer(bool newaddr); // glBuffer(Sub)Data calls

  const limnPolyData *_lpld;  // cannot limnPolyDataNix()
  limnPolyData *_lpldOwn;     //   can  limnPolyDataNix()
  /* stores a shallow copy of lpld, so that rebuffer can tell value of
     "newaddr" for _buffer() */
  limnPolyData _lpldCopy;

  /* management of GL buffers for the xyzw and the limnPolyDataInfo */
  unsigned int _buffNum;
  /* the GL buffers; allocated for buffNum */
  GLuint *_buff;
  /* map from Hale::vertAttrIdx into _buff */
  int _buffIdx[HALE_VERT_ATTR_IDX_NUM];
  /* GL element array buffer */
  GLuint _elms;
  /* the program used for rendering */
  const Program *_program;

  std::vector<GLuint> _textureIds;
  std::unordered_map<char*, unsigned int> _textureVars;
};

/*
** Right now this is just sort of a disorganized bag of state, off of which
** we can hang things that would otherwise be per-(C)program globals, and
** just to collect stuff that is in the process of being figured out (so it
** is sort of like a HaleContext; For now, the Scene owns nothing dynamically allocated.
**
** One principle is that the Scene is oblivious to the Viewer: so the light
** direction here is in world-space, not view-space
*/
class Scene {
 public:
  explicit Scene();
  ~Scene();

  void add(const Polydata *pd);

  /* set/get background color */
  void bgColor(float rr, float gg, float bb);
  glm::vec3 bgColor(void) const;

  /* set/get world-space light direction */
  void lightDir(glm::vec3 dir);
  glm::vec3 lightDir(void) const;

  void bounds(glm::vec3 &min, glm::vec3 &max) const;

  void drawInit(void);
  void draw(void);
 protected:
  float _bgColor[3];
  glm::vec3 _lightDir;
  std::list<const Polydata *> _polydata;
};

} // namespace Hale

#endif /* HALE_INCLUDED */
