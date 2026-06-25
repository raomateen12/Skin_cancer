import Navbar from "@/components/landing/Navbar";
import HeroSection from "@/components/landing/HeroSection";
import HowItWorksSection from "@/components/landing/HowItWorksSection";
import CapabilitiesSection from "@/components/landing/CapabilitiesSection";
import SafetySection from "@/components/landing/SafetySection";
import Footer from "@/components/landing/Footer";

export default function LandingPage() {
  return (
    <main>
      <Navbar />
      <HeroSection />
      <HowItWorksSection />
      <CapabilitiesSection />
      <SafetySection />
      <Footer />
    </main>
  );
}
