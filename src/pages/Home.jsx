import { motion } from 'motion/react';
import { Target, Layers, Zap, ArrowRight, Cpu, ShieldAlert, BadgeInfo } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import ThreeDScanner from '../components/ThreeDScanner';
import { cn } from '../lib/utils';

export default function Home() {
  const navigate = useNavigate();

  const bentoItems = [
    { 
      icon: <Cpu className="w-6 h-6" />, 
      title: "Neural Extraction", 
      desc: "Deep-learning models identify ridge features at sub-pixel resolution.",
      className: "bg-muted/50"
    },
    { 
      icon: <Target className="w-6 h-6" />, 
      title: "Precision Alignment", 
      desc: "Automated orientation and scaling for biometric standardization.",
      className: "bg-secondary text-secondary-foreground"
    },
    { 
      icon: <Layers className="w-6 h-6" />, 
      title: "Multi-modal Fusion", 
      desc: "Synthesizing multiple scan passes into a high-fidelity singular record.",
      className: "bg-secondary text-secondary-foreground"
    },
    { 
      icon: <Zap className="w-6 h-6" />, 
      title: "Encrypted Latency", 
      desc: "All processing occurs in an air-gapped, zero-trust cloud environment.",
      className: "bg-primary text-primary-foreground"
    }
  ];

  return (
    <div className="overflow-hidden bg-background">
      {/* Hero Section with Video Background */}
      <section className="relative min-h-[90vh] flex items-center pt-32 pb-20">
        <div className="absolute inset-0 z-0 overflow-hidden">
          <video 
            autoPlay 
            loop 
            muted 
            playsInline
            className="bg-video absolute top-1/2 left-1/2 w-auto min-w-full min-h-full max-w-none -translate-x-1/2 -translate-y-1/2 object-cover"
          >
            <source src="/bg-video.mp4" type="video/mp4" />
          </video>
          {/* Light overlay to improve text visibility */}
          <div className="absolute inset-0 bg-white/70 backdrop-blur-[2px] z-10" />
        </div>
        
        <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 lg:grid-cols-2 gap-16 items-center relative z-20">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-border text-secondary text-[10px] font-bold mb-8 tracking-[0.2em] uppercase shadow-sm">
              <Zap className="w-3.5 h-3.5" /> Intelligence at the edge
            </div>
            <h1 className="text-6xl md:text-8xl font-black tracking-tight mb-8 leading-[0.9] text-foreground">
              BIOMETRIC <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-secondary to-primary">INSIGHT.</span>
            </h1>
            <p className="text-xl text-foreground/80 mb-10 max-w-xl leading-relaxed font-medium">
              Enhance biometric accuracy for everyone. Our system categorizes normal, aged, and labour fingerprints, applying specialized enhancement to damaged fingerprints to ensure reliable and fast identity verification. 
            </p>
            <div className="flex flex-wrap gap-4">
              <button 
                onClick={() => navigate('/scanner')}
                className="bg-primary text-primary-foreground px-10 py-5 rounded-2xl font-bold flex items-center gap-4 hover:scale-[1.02] transition-all premium-shadow group"
              >
                Get Started <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1.2, delay: 0.2 }}
            className="h-[500px] lg:h-[650px] relative hidden lg:flex flex-col items-center justify-center p-8 bg-transparent"
          >
            <div className="relative z-10 w-full flex-grow">
               <ThreeDScanner />
            </div>
          </motion.div>
        </div>
      </section>

      {/* Problem Statement Section */}
      <section className="py-32 bg-muted/30">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-20">
             <h2 className="text-sm font-black tracking-[0.3em] text-secondary uppercase mb-4">The Real-World Challenge</h2>
             <h3 className="text-4xl md:text-5xl font-extrabold tracking-tighter">Why standard verification fails.</h3>
             <p className="mt-4 text-muted-foreground text-lg max-w-2xl mx-auto">
              Traditional biometric systems struggle with the realities of aging and labour, often rejecting faded, cut, or worn fingerprints
             </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
             <motion.div 
               initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
               className="bg-card p-10 rounded-3xl border premium-shadow relative overflow-hidden"
             >
                <div className="w-14 h-14 bg-red-100 dark:bg-red-900/20 rounded-2xl flex items-center justify-center mb-6">
                   <BadgeInfo className="w-7 h-7 text-red-600" />
                </div>
                <h4 className="text-2xl font-bold mb-4">The Real-World Challenge</h4>
                <ul className="space-y-3 text-muted-foreground font-medium">
                   <li className="flex items-start gap-2">
                     <div className="w-1.5 h-1.5 rounded-full  mt-2"></div>
                     <span>Wrinkly, cut, scarred, or worn fingerprints common in labor-intensive work. Example: A worker with rough hands is unable to unlock a system.</span>
                   </li>
                   <li className="flex items-start gap-2">
                     <div className="w-1.5 h-1.5 rounded-full  mt-2"></div>
                     <span>Aging effects on fingerprint patterns naturally over time. Fingerprints fades, causing scanner to reject or misread the fingerprints</span>
                   </li>
                   <li className="flex items-start gap-2">
                     <div className="w-1.5 h-1.5 rounded-full  mt-2"></div>
                   </li>
                </ul>
             </motion.div>

             <motion.div 
               initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.1 }}
               className="bg-card p-10 rounded-3xl border premium-shadow relative overflow-hidden"
             >
                <div className="w-14 h-14 bg-orange-100 dark:bg-orange-900/20 rounded-2xl flex items-center justify-center mb-6">
                   <Layers className="w-7 h-7 text-orange-600" />
                </div>
                <h4 className="text-2xl font-bold mb-4">System-Related Problems</h4>
                <ul className="space-y-3 text-muted-foreground font-medium">
                   <li className="flex items-start gap-2">
                     <div className="w-1.5 h-1.5 rounded-full  mt-2"></div>
                     <span>Wrong biometric data records causing false negatives.</span>
                   </li>
                   <li className="flex items-start gap-2">
                     <div className="w-1.5 h-1.5 rounded-full mt-2"></div>
                     <span>Poor template matching algorithms degrading reliability.</span>
                   </li>
                   <li className="flex items-start gap-2">
                     <div className="w-1.5 h-1.5 rounded-full mt-2"></div>
                     <span>Large databases causing extremely slow search and matching performance.</span>
                   </li>
                </ul>
             </motion.div>

             <motion.div 
               initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.2 }}
               className="bg-card p-10 rounded-3xl border premium-shadow relative overflow-hidden"
             >
                <div className="w-14 h-14 bg-orange-100 dark:bg-orange-900/20 rounded-2xl flex items-center justify-center mb-6">
                   <ShieldAlert className="w-7 h-7 text-secondary text-orange-600" />
                </div>
                <h4 className="text-2xl font-bold mb-4 ">Integrity Risks</h4>
                <ul className="space-y-3 text-muted-foreground font-medium">
                   <li className="flex items-start gap-2">
                     <div className="w-1.5 h-1.5 rounded-full bg-secondary mt-2"></div>
                     <span>Spoofing vulnerabilities (fake fingerprints, photos, masks). Example: An attacker uses a fake fingerprint mold to bypass authentication.</span>
                   </li>
                   <li className="flex items-start gap-2">
                     <div className="w-1.5 h-1.5 rounded-full bg-secondary mt-2"></div>
                     <span>Data breaches resulting in irreversible biometric identity theft.</span>
                   </li>
                   <li className="flex items-start gap-2">
                     <div className="w-1.5 h-1.5 rounded-full bg-secondary mt-2"></div>
                     <span>Replay attacks undermining system trust protocols.</span>
                   </li>
                </ul>
             </motion.div>
          </div>
        </div>
      </section>

      {/* Feature Grid */}
      <section className="py-32 border-t border-border">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-20">
            <h2 className="text-sm font-black tracking-[0.3em] text-secondary uppercase mb-4">Core Technology</h2>
            <h3 className="text-4xl md:text-5xl font-extrabold tracking-tighter">System Overview</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {bentoItems.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className={cn(
                  "p-6 rounded-[1.5rem] border flex flex-col justify-between transition-all hover:border-secondary/50 hover:premium-shadow group",
                  item.className
                )}
              >
                <div className="mb-4 p-3 bg-background w-fit rounded-xl group-hover:scale-110 transition-transform shadow-sm text-foreground">
                  {item.icon}
                </div>
                <div>
                  <h4 className="text-lg font-bold mb-2 tracking-tight">{item.title}</h4>
                  <p className="text-current opacity-90 text-sm font-medium leading-relaxed">{item.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
