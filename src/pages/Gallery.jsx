import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, Fingerprint, Scan, Maximize2, Info } from 'lucide-react';
import { cn } from '../lib/utils';

export default function Gallery() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [feedback, setFeedback] = useState("");

  // Updated images specifically focused on biometric/fingerprint scanning failures and corrupted data
  const images = [
    { id: 1, src: "https://en.dailypakistan.com.pk/wp-content/uploads/2024/12/sindh-govt-extends-biometric-deadline-for-old-vehicles-by-two-months-news-262402-850x478.jpg" },
    { id: 2, src: "https://www.geo.tv/assets/uploads/updates/2021-01-17/330350_3512821_updates.jpg" },
    { id: 3, src: "https://plus.unsplash.com/premium_photo-1758523722005-c125eab5d0f5?q=80&w=870&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" },
    { id: 4, src: "https://media.istockphoto.com/id/1441570156/photo/closeup-shot-of-a-human-finger-skin-on-a-black-background.jpg?b=1&s=170x170&k=20&c=Hr5XG_P-gAbw1IgDyzpZ9wHrNnm6IuSgwe0YpUBlmyM=" },
    { id: 5, src: "https://images.unsplash.com/photo-1611330500121-d9439ddc3d9d?q=80&w=876&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" },
    { id: 6, src: "https://img-cdn.inc.com/image/upload/f_webp,c_fit,w_828,q_auto/images/panoramic/getty_528296984_200012462000928074_232561.jpg" },
  ];

  return (
    <div className="pt-40 pb-32 bg-background min-h-screen">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-24"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-secondary/20 text-secondary text-[10px] font-bold mb-6 tracking-widest uppercase">
            <Maximize2 className="w-4 h-4" /> Neural Assets
          </div>
          <h1 className="text-5xl md:text-7xl font-black mb-6 tracking-tighter leading-none text-foreground">
            Digital <span className="text-transparent bg-clip-text bg-gradient-to-r from-secondary to-primary">Forensics.</span>
          </h1>
          <p className="text-xl text-muted-foreground font-medium max-w-2xl leading-relaxed">
            Exploring the visual frontier of biometric identity failures, corrupted databases, and degraded hardware mapping patterns.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {images.map((image) => (
            <motion.div
              key={image.id}
              layoutId={`image-${image.id}`}
              onClick={() => setSelectedImage(image)}
              className="relative aspect-square rounded-[2rem] overflow-hidden cursor-pointer group border border-border bg-muted/50"
            >
              <img
                src={image.src}
                alt={image.title}
                className="w-full h-full object-cover transition-all duration-700 group-hover:scale-105 group-hover:blur-[2px]"
                referrerPolicy="no-referrer"
              />
              <div className="absolute inset-0 bg-primary/30 opacity-0 group-hover:opacity-100 transition-all duration-500 flex flex-col items-center justify-center">
                <div className="p-4 bg-background dark:bg-background/90 rounded-2xl premium-shadow border scale-90 group-hover:scale-100 transition-transform">
                  <Scan className="w-6 h-6 text-foreground" />
                </div>
              </div>
              <div className="absolute bottom-0 left-0 right-0 p-8 bg-gradient-to-t from-black/90 via-black/50 to-transparent translate-y-4 group-hover:translate-y-0 transition-transform duration-500">
                <span className="text-[10px] font-black tracking-widest text-secondary uppercase mb-1 block">{image.category}</span>
                <p className="text-white font-bold text-lg tracking-tight">{image.title}</p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* User Feedback Section */}
        <motion.div
           initial={{ opacity: 0, y: 30 }}
           whileInView={{ opacity: 1, y: 0 }}
           viewport={{ once: true }}
           className="mt-32 max-w-2xl mx-auto"
        >
           <h3 className="text-2xl font-black tracking-tight mb-4 text-center">Analyst Feedback</h3>
           <p className="text-muted-foreground text-center mb-6">
             Notice any inconsistencies in the forensic data? Provide your input below.
           </p>
           <div className="bg-card p-6 rounded-3xl border premium-shadow relative z-10">
              <textarea
                 value={feedback}
                 onChange={(e) => setFeedback(e.target.value)}
                 id="feedback-input"
                 placeholder="Enter observations on biometric degradation..."
                 className="w-full h-32 p-4 bg-background border rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-secondary/50 mb-4 font-medium placeholder:text-muted-foreground/60 transition-all"
              />
              <button 
                onClick={() => setFeedback('')}
                className="w-full bg-primary text-primary-foreground py-3 rounded-xl font-bold flex items-center justify-center gap-2 hover:opacity-90 transition-opacity"
              >
                 Submit Feedback
              </button>
           </div>
        </motion.div>
      </div>

      <AnimatePresence>
        {selectedImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-background/80 backdrop-blur-xl flex items-center justify-center p-4 md:p-8 lg:p-12"
            onClick={() => setSelectedImage(null)}
          >
            <motion.div
              layoutId={`image-${selectedImage.id}`}
              className="relative max-w-6xl w-full bg-background/95 border dark:border-border/50 rounded-[2rem] overflow-hidden shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="grid grid-cols-1 lg:grid-cols-3 h-full">
                <div className="lg:col-span-2 aspect-[4/3] lg:aspect-auto h-full max-h-[60vh] lg:max-h-none bg-muted flex items-center justify-center relative overflow-hidden">
                  <img
                    src={selectedImage.src}
                    alt={selectedImage.title}
                    className="w-full h-full object-cover"
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute inset-0 bg-blue-500/10 mix-blend-overlay pointer-events-none" />
                </div>
                <div className="p-8 lg:p-10 flex flex-col justify-between bg-card text-card-foreground">
                  <div className="space-y-8">
                    <div className="flex justify-between items-start">
                      <div>
                        <span className="text-[10px] font-black tracking-widest text-brand-500 uppercase">{selectedImage.category}</span>
                        <h3 className="text-3xl font-black tracking-tighter mt-2">{selectedImage.title}</h3>
                      </div>
                      <button
                        onClick={() => setSelectedImage(null)}
                        className="p-3 bg-secondary hover:bg-muted text-secondary-foreground rounded-2xl transition-colors"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </div>
                    
                    <div className="space-y-4">
                      <p className="text-muted-foreground font-medium leading-relaxed text-sm">
                        High-resolution capture from the NeuralScan core research facility focusing on biometric identification degradation mapping. Demonstrates the precision of Biometric Insight interpretation in edge cases.
                      </p>
                    </div>
                  </div>

                  <div className="pt-8 flex gap-3">
                    <button className="flex-1 p-4 bg-primary text-primary-foreground rounded-xl flex items-center justify-center gap-2 font-bold text-sm hover:opacity-90 transition-opacity">
                      <Scan className="w-4 h-4" /> Analyse Biometrics
                    </button>
                    <button className="p-4 bg-secondary text-secondary-foreground rounded-xl hover:bg-muted transition-colors">
                      <Fingerprint className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
